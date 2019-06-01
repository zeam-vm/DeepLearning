Deep Learning with elixir
This project is called Deep Pipe(DP)

Network example (See test.ex)

defnetwork init_network2(_x) do
  _x |> f(5,5) |> flatten
  |> w(576,100) |> b(100) |> sigmoid
  |> w(100,10) |> b(10) |> sigmoid
end

Usage:
  iex -S mix

module DP is Deep Pipe(DP) module
module DPB is DP for batch
module DPP is DP for parallel
module Tensor is code for CNN data operation
module Dmatrix is code for Matrix
module Pmatrix is code for Matrix product in paralell
module MNIST is code for MNIST data set

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
filter {:filter,w,lr,st,v} st is strut for convolution
pad    {:pad,n} n is size of padding
pool   {:pool,st} st is strut
function {:function,f,g} f is function, g is differential function
softmax {:softmax,f,_} f is function, only output layer softmax is set with cross_entropy

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

gradient/3 caluculate gradient by backpropagation
gradient(x,network,t)  x is data, t is train data
learning/2, /3 update network with gradient
learning(network,gradient)  update with sgd
learning(network,gradient,:momentum) update with momentum method
learning(network,gradient,:adagrad) update with adagrad method
learning(network,gradient,:adam) update with adam method

module DPP
gradient/3 caluculate gradient by backpropagation in paralell
gradient(x,network,t)  x is data, t is train data, divide size is default(5)
gradient(x,network,t,d)  d is dvide size
