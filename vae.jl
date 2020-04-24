# import Automatic Differentiation
# You may use Neural Network Framework, but only for building MLPs
# i.e. no fancy probabilistic implementations
using Flux
using MLDatasets
using Statistics
using Logging
using Test
using Random
using StatsFuns: log1pexp
using Zygote
using Distributions
using Logging
Random.seed!(412414);

#### Probability Stuff
# Make sure you test these against a standard implementation!

# log-pdf of x under Factorized or Diagonal Gaussian N(x|μ,σI)
function factorized_gaussian_log_density(mu, logsig, xs)
  """
  mu and logsig either same size as x in batch or same as whole batch
  returns a 1 x batchsize array of likelihoods
  """
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((xs .- mu).^2)./(σ.^2),dims=1)
end;

# log-pdf of x under Bernoulli
function bernoulli_log_density(logit_means,x)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
  b = x .* 2 .- 1 # {0,1} -> {-1,1}
  return - log1pexp.(-b .* logit_means)
end;
## This is really bernoulli
@testset "test stable bernoulli" begin
  using Distributions
  x = rand(10,100) .> 0.5
  μ = rand(10)
  logit_μ = log.(μ./(1 .- μ))
  @test logpdf.(Bernoulli.(μ),x) ≈ bernoulli_log_density(logit_μ,x)
  # over i.i.d. batch
  @test sum(logpdf.(Bernoulli.(μ),x),dims=1) ≈ sum(bernoulli_log_density(logit_μ,x),dims=1)
end;

# sample from Diagonal Gaussian x~N(μ,σI) (hint: use reparameterization trick here)
sample_diag_gaussian(μ,logσ) = (ϵ = randn(size(μ)); μ .+ exp.(logσ).*ϵ);
# sample from Bernoulli (this can just be supplied by library)
sample_bernoulli(θ) = rand.(Bernoulli.(θ));

# Load MNIST data, binarise it, split into train and test sets (10000 each) and partition train into mini-batches of M=100.
# You may use the utilities from A2, or dataloaders provided by a framework
function load_binarized_mnist(train_size=1000, test_size=1000)
  train_x, train_label = MNIST.traindata(1:train_size);
  test_x, test_label = MNIST.testdata(1:test_size);
  @info "Loaded MNIST digits with dimensionality $(size(train_x))"
  train_x = reshape(train_x, 28*28,:)
  test_x = reshape(test_x, 28*28,:)
  @info "Reshaped MNIST digits to vectors, dimensionality $(size(train_x))"
  train_x = train_x .> 0.5; #binarize
  test_x = test_x .> 0.5; #binarize
  @info "Binarized the pixels"
  return (train_x, train_label), (test_x, test_label)
end;

function batch_data((x,label)::Tuple, batch_size=100)
  """
  Shuffle both data and image and put into batches
  """
  N = size(x)[end] # number of examples in set
  rand_idx = shuffle(1:N) # randomly shuffle batch elements
  batch_idx = Iterators.partition(rand_idx,batch_size) # split into batches
  batch_x = [x[:,i] for i in batch_idx]
  batch_label = [label[i] for i in batch_idx]
  return zip(batch_x, batch_label)
end;
# if you only want to batch xs
batch_x(x::AbstractArray, batch_size=100) = first.(batch_data((x,zeros(size(x)[end])),batch_size));


### Implementing the model

## Load the Data
train_data, test_data = load_binarized_mnist();
train_x, train_label = train_data;
test_x, test_label = test_data;

## Test the dimensions of loaded data
@testset "correct dimensions" begin
  @test size(train_x) == (784,1000)
  @test size(train_label) == (1000,)
  @test size(test_x) == (784,1000)
  @test size(test_label) == (1000,)
end;

## Model Dimensionality
# #### Set up model according to Appendix C (using Bernoulli decoder for Binarized MNIST)
# Set latent dimensionality=2 and number of hidden units=500.
Dz, Dh = 2, 500;
Ddata = 28^2;

# ## Generative Model
# This will require implementing a simple MLP neural network
# See example_flux_model.jl for inspiration
# Further, you should read the Basics section of the Flux.jl documentation
# https://fluxml.ai/Flux.jl/stable/models/basics/
# that goes over the simple functions you will use.
# You will see that there's nothing magical going on inside these neural network libraries
# and when you implemented a neural network in previous assignments you did most of the work.
# If you want more information about how to use the functions from Flux, you can always reference
# the internal docs for each function by typing `?` into the REPL:
# ? Chain
# ? Dense

# Q1(a) log_prior(z)
function log_prior(z)
  return factorized_gaussian_log_density(0, 0, z)
end;

# Q1(b) decoder
decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata));

# Q1(c)
function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z)"""
  # parameters decoded from latent z
  logitμ = decoder(z)
  # return likelihood for each element in batch
  return  sum(bernoulli_log_density(logitμ,x), dims=1)
end;

# Q1(d)
joint_log_density(x,z) = log_prior(z) .+ log_likelihood(x,z);

## Amortized Inference
# Q2(a)
function unpack_gaussian_params(θ)
  μ, logσ = θ[1:2,:], θ[3:end,:]
  return  μ, logσ
end;


encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz*2), unpack_gaussian_params);
# Hint: last "layer" in Chain can be 'unpack_gaussian_params'

# Q2(b)
# write log likelihood under variational distribution.
log_q(q_μ, q_logσ, z) = factorized_gaussian_log_density(q_μ, q_logσ, z);

# Q2(c)
function elbo(x)
  q_μ, q_logσ = encoder(x) # variational parameters from data
  z = sample_diag_gaussian(q_μ, q_logσ) # sample from variational distribution
  joint_ll = joint_log_density(x,z) # joint likelihood of z and x under model
  log_q_z = log_q(q_μ, q_logσ, z) # likelihood of z under variational distribution
  elbo_estimate = mean(joint_ll - log_q_z) # Scalar value, mean variational evidence lower bound over batch
  return elbo_estimate
end;

# Q2(d)
function loss(x)
  return -elbo(x) # scalar value for the variational loss over elements in the batch
end;

# Q2(e)
# Training with gradient optimization:
# See example_flux_model.jl for inspiration

function train_model_params!(loss, encoder, decoder, train_x, test_x; nepochs=10)
  # model params
  ps = Flux.params(encoder,decoder) # parameters to update with gradient descent
  # ADAM optimizer with default parameters
  opt = ADAM()
  # over batches of the data
  for i in 1:nepochs
    for d in batch_x(train_x)
      gs = Flux.gradient(ps) do # compute gradients with respect to variational loss over batch
        batch_loss = loss(d)
        return batch_loss
      end
    Flux.Optimise.update!(opt,ps,gs) #update the paramters with gradients
    end
    if i%10 == 0 # change 1 to higher number to compute and print less frequently
      @info "Test loss at epoch $i: $(loss(batch_x(test_x)[1]))"
    end
  end
  @info "Parameters of encoder and decoder trained!"
end;

## Train the model
train_model_params!(loss,encoder,decoder,train_x,test_x, nepochs=100);

### Save the trained model!
# using BSON:@save
# cd(@__DIR__)
# @info "Changed directory to $(@__DIR__)"
# save_dir = "trained_models"
# if !(isdir(save_dir))
#   mkdir(save_dir)
#   @info "Created save directory $save_dir"
# end
# @save joinpath(save_dir,"encoder_params.bson") encoder
# @save joinpath(save_dir,"decoder_params.bson") decoder
# @info "Saved model params in $save_dir"



## Load the trained model!
using BSON:@load
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
load_dir = "trained_models"
@load joinpath(load_dir,"encoder_params.bson") encoder
@load joinpath(load_dir,"decoder_params.bson") decoder
@info "Load model params from $load_dir"


# Visualization
using Images
using Plots
# make vector of digits into images, works on batches also
mnist_img(x) = ndims(x)==2 ? Gray.(reshape(x,28,28,:)) : Gray.(reshape(x,28,28))

## Example for how to use mnist_img to plot digit from training data
plot(mnist_img(train_x[:,3]))

# Q3(a)
plot1_list = Any[];
plot2_list = Any[];

for i in 1:10
  z = sample_diag_gaussian([0,0],[0,0]);
  logitμ = decoder(z);
  prob = exp.(logitμ) ./ (1 .+ exp.(logitμ));
  prob_plot = plot(mnist_img(prob));
  binary = sample_bernoulli(prob);
  binary_plot = plot(mnist_img(binary));
  push!(plot1_list, prob_plot);
  push!(plot2_list, binary_plot);
end;

plot_list = hcat(plot1_list, plot2_list);
display(plot(plot_list..., layout = grid(2, 10), size=(7840,784*2)))
# savefig("Q3_a")

# Q3(b)
latent = encoder(train_x);
prob_mean = latent[1];
x, y = prob_mean[1,:], prob_mean[2,:];
scatter(x,y, group=train_label, legend=:topleft)
# savefig("Q3_b")

# Q3(c)
function interpolation(za, zb, α)
  return za*α+zb*(1-α)
end;

image1 = train_x[:,5]; # label=9
image2 = train_x[:,6]; # label=2
image3 = train_x[:,4]; # label=1

logit1 = decoder(encoder(image1)[1]);
mean1 = exp.(logit1) ./ (1 .+ exp.(logit1));

logit2 = decoder(encoder(image2)[1]);
mean2 = exp.(logit2) ./ (1 .+ exp.(logit2));

logit3 = decoder(encoder(image3)[1]);
mean3 = exp.(logit3) ./ (1 .+ exp.(logit3));

plot12_list = Any[];
for i in 0:9
  α = i/10;
  mean12 = interpolation.(mean1, mean2, α);
  mean12_plot = plot(mnist_img(vec(mean12)));
  push!(plot12_list, mean12_plot);
end

plot13_list = Any[];
for i in 0:9
  α = i/10;
  mean13 = interpolation.(mean1, mean3, α);
  mean13_plot = plot(mnist_img(vec(mean13)));
  push!(plot13_list, mean13_plot);
end

plot23_list = Any[];
for i in 0:9
  α = i/10;
  mean23 = interpolation.(mean2, mean3, α);
  mean23_plot = plot(mnist_img(vec(mean23)));
  push!(plot23_list, mean23_plot);
end

polt_list = [plot12_list; plot13_list; plot23_list];
display(plot(polt_list..., layout = grid(3, 10), size=(7840,784*3)))
savefig("Q3_c")

# Q4(a)
function top_image(x)
  # Dx, Db = size(x)
  x = reshape(x,28,28)
  topx = x[1:14,:]
  pad = zeros(14,28)
  newx = [topx; pad]
  return vec(newx)
end;

function bottom_image(x)
  x = reshape(x,28,28)
  bottomx = x[15:28,:]
  pad = zeros(14,28)
  newx = [pad; bottomx]
  return vec(newx)
end;

function logp_top_given_z(x, z)
  """ Compute log likelihood log_p(topx|z)"""
  logitμ = decoder(z)
  logitμ_top = top_image(logitμ)
  x_top = top_image(x)
  return sum(bernoulli_log_density(logitμ_top, x_top), dims=1)
end;

function joint_log_topx_z(x, z)
  """ Compute log likelihood log_p(topx,z)"""
  prior = log_prior(z)
  cond = logp_top_given_z(x,z)
  joint = prior + cond
  return joint
end;

# Q4(b)_(a)
x = train_x[:,6];
μ_init = [0.5, 0.5];
logσ_init = [0, 0];
init_params = (μ_init, logσ_init);

function sample_z(params)
  z_μ = params[1]
  z_logσ = params[2]
  z = sample_diag_gaussian(z_μ,z_logσ)
  return z
end;

# Q4(b)_(b)
function elbo_half(params, x, z)
  z_μ = params[1]
  z_logσ = params[2]
  logp_joint = joint_log_topx_z(x, z)
  logq = factorized_gaussian_log_density(z_μ, z_logσ, z)
  return mean(logp_joint .- logq)
end;

# Q4(b)_(c)
function fit_variational_dist(init_params, x; num_itrs=10000, lr= 1e-4)
  params_cur = init_params
  for i in 1:num_itrs
    z = sample_z(params_cur)
    loss(params) = -elbo_half(params, x, z)
    grad_params = gradient(loss, params_cur)[1]
    μ_grad = grad_params[1]
    logσ_grad = grad_params[2]
    μ = params_cur[1] - lr .* μ_grad
    logσ = params_cur[2] - lr .* logσ_grad
    params_cur = (μ, logσ)

    if i%100 == 0 # change 1 to higher number to compute and print less frequently
      @info "Test loss at epoch $i: $(loss(params_cur))"
    end
  end
  return params_cur
end;

params = fit_variational_dist(init_params, x);

# Q4(b)_(d)
function skillcontour!(f; colour=nothing)
  n = 100
  x = range(-3,stop=3,length=n)
  y = range(-3,stop=3,length=n)
  z_grid = Iterators.product(x,y) # meshgrid for contour
  z_grid = reshape.(collect.(z_grid),:,1) # add single batch dim
  z = f.(z_grid)
  z = getindex.(z,1)'
  max_z = maximum(z)
  levels = [.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2] .* max_z
  if colour==nothing
  p1 = contour!(x, y, z, fill=false, levels=levels)
  else
  p1 = contour!(x, y, z, fill=false, c=colour,levels=levels,colorbar=false)
  end
  plot!(p1)
end;

p(zs) = exp.(joint_log_topx_z(x, zs));
q(zs) = exp.(factorized_gaussian_log_density(params[1], params[2], zs));
plot(title="Compare p & q",
    xlabel = "mean1",
    ylabel = "mean2"
   )
display(skillcontour!(p; colour=:red))
display(skillcontour!(q; colour=:blue))
# savefig("Q4_b_d")

# Q4(b)_(e)
mnist_half(x) = ndims(x)==2 ? Gray.(reshape(x,28,14,:)) : Gray.(transpose(reshape(x,28,14)));
sample = sample_diag_gaussian(params[1], params[2]);
logit_sample = decoder(sample);
prob_sample = exp.(logit_sample) ./ (1 .+ exp.(logit_sample));
whole_plot = top_image(train_x[:,6]) + bottom_image(prob_sample);
plot(mnist_img(whole_plot))
plot(mnist_img(train_x[:,6]))
# savefig("Q4_b_e2")
