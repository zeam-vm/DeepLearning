For my study Deep Learning with elixir
This project is called Function Flow(FF)

Network example

defnetwork init_network2(_x) do
  _x |> f(5,5,0.3,1,0.05) |> flatten
  |> w(576,100) |> b(100) |> sigmoid
  |> w(100,10) |> b(10) |> sigmoid
end

Usage:
  iex -S mix

module FF is Function Flow(FF) module
module FFB is FF for batch
module Tensor is code for CNN data operation
module Dmatrix is code for Matrix
module Pmatrix is code for Matrix product in paralell
module MNIST is code for MNIST data set

I implemented backpropagation and numerical-gradient
Now I'm testing small data set.

I implemented CNN and testing.
New frame work is called Function Flow (FF)
module FF
This module is extremely incomplete.Now I am improving

expample:
iex(3)> FF.online(1000,10)
preparing data
0.91480261503261
0.7066400905106333
0.49963843981861683
0.5397736265997177
0.13208250353334347
0.13669230966971024
0.07858477636565094
0.48540650676689406
0.001027589605178186
0.1712154333334797
accuracy rate =
0.88
:ok
