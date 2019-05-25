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
