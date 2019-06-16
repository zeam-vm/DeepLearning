Deep Learning with elixir
This project is called Deep Pipe(DP)

Network example (See test.ex)

defnetwork init_network2(_x) do
  _x |> f(5,5) |> flatten
  |> w(576,100) |> b(100) |> sigmoid
  |> w(100,10) |> b(10) |> sigmoid
end

Install:
  sudo apt install build-essential
  sudo apt-get install build-essential erlang-dev libatlas-base-dev
  mix deps.get


Usage:
  iex -S mix

module DP is Deep Pipe(DP) module
module DPB is DP for batch
module DPP is DP for parallel
module CTensor is code for CNN data operation
module Cmatrix is code for Matrix
module MNIST is code for MNIST data set
module Time is time/1 for measure execution time

I implemented backpropagation and numerical-gradient
Now I'm testing small data set.


expample:
iex(1)> require Time

Time
iex(2)> Time.time(Test.adagrad(100,50))
preparing data
ready
0.44383196477296905
0.37511510344740406
0.42960276053222174
0.352539961358792
0.2861907950783934
0.21772105559847485
0.1880808136708525
0.14605224305760664
...
0.016682469588708566
0.019254450344041836
0.00594231528389093
0.013773451908515
0.019834342678945693
accuracy rate = 0.88
"time: 202819950 micro second"
"-------------"
:ok

>

specification:

data structure
matrix  e.g. [[1,2],[3,4]]
row_vector e.g. [[1,2]]
tensor e.g. [[[1,2],[3,4]],[[5,6],[7,8]]]
network e.g. [{:weight,w,lr,v},{:bias,b,lr},{:function,f,g}]
weight {:weight,w,lr,v} w is matrix, lr is learning rate, v is for momentum,adagrad,adam
bias   {:bias,b,lr,v} b is row vector
filter {:filter,w,lr,st,v} st is stradd for convolution
pad    {:pad,n} n is size of padding
pool   {:pool,st} st is stradd
function {:function,f,g} f is function, g is differential function
softmax {:softmax,f,_} f is function, only output layer softmax is set with cross_entropy

module macros
defnetwork is macros to describe network
argument must have under bar to avoid warning message

w(m,n)  weight matrix size(m,n). elements are Gaussian distribution random float
w(m,n,lr) lr is learning rate (default is 0.1)
w(m,n,lr,z) z is multiple for Gaussian distribution random float. (default is 0.1)

b(n) bias row_vector size(n). elements are all zero
b(n,lr) lr is learning rate (default is 0.1)

function sigmoid,relu,ident,softmax

f(m,n) filter matrix size(m,n). elements are Gaussian distribution random float
f(m,n,lr) lr is learning rate
f(m,n,lr,z) z is multiple for Gaussian distribution random float.(default is 0.1)
f(m,n,lr,z,st) st is stradd

pad(n) padding n is size of padding

pool(st) pooling stradd size is st

module DP
forward/2 forward calculation for single data
forward(x,network) x is data(row_vector) , network
numerical_gradient/3 calculate gradient by numerical differentiation
numerical_gradient(x,network,t) x is data, t is train data, loss function is mean_square
numerical_gradient(x,network,t,:cross) loss function is cross_entropy

gradient/3 caluculate gradient by backpropagation
gradient(x,network,t)  x is data, t is train data
learning/2, /3 update network with gradient
learning(network,gradient)  update with sgd
learning(network,gradient,:momentum) update with momentum method
learning(network,gradient,:adagrad) update with adagrad method
learning(network,gradient,:adam) update with adam method

print/1 print data
newline/0 print LF
save/2 save network data to file
save(filename,network)
load/1 load network data from file
load(filename)

mean_square/2 loss function
mean_square(y,t) y is row vector of result and t is row vector of train
cross_entropy/2 loss function
cross_entropy(y,t) y is row vector of result and t is row vector of train


module DPB
all data is tensor
forward/2 forward calculation for batch data
forward(x,network) x is data(row_vector) , network

numerical_gradient/3 calculate gradient by numerical differentiation
numerical_gradient(x,network,t) x is data, t is train data, loss function is mean_square
numerical_gradient(x,network,t,:cross) loss function is cross_entropy
numerical_gradient(x,network,t,:square) loss function is mean_square

gradient/3 caluculate gradient by backpropagation
gradient(x,network,t)  x is data, t is train data

learning/2, /3 update network with gradient
learning(network,gradient)  update with sgd
learning(network,gradient,:sgd)
learning(network,gradient,:momentum) update with momentum method
learning(network,gradient,:adagrad) update with adagrad method
learning(network,gradient,:adam) update with adam method

loss(y,t,:cross) calculate loss with cross_entropy. y is result data,t is train data
loss(y,t,:square) calculate loss with mean_square. y is result data,t is train data

accuracy/3  calculate accuracy
accuracy(image,network,label) image is test image data, label is test label (onehot)

module DPP
gradient/3 caluculate gradient by backpropagation in paralell
gradient(x,network,t)  x is data, t is train data, divide size is default(5)
gradient(x,network,t,d)  d is dvide size

module MNIST
train_image(n)  get train image data size of n. Each data is 28*28 matrix
train_image(n,:flatten) get train image data size of n. Each data is 784 row_vector
train_label_onehot(n) get train label data aas onehot row_vector

test_image(n)  get test image data size of n. Each data is 28*28 matrix
test_image(n,:flatten) get test image data size of n. Each data is 784 row_vector
test_label_onehot(n) get test label data aas onehot row_vector

module time
time/1
time(func) e.g. time(1+2)
