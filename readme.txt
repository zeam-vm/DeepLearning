For my study Deep Learning with elixir
This project is called Function Flow(FF)

Network example

defnetwork init_network2(_x) do
  _x |> f(5,5) |> flatten
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
> require Time
> Time.time(FFB.sgd(20,200))
preparing data
ready
0.4474500733730803
0.46611457228999553
...
0.010044398231206475
0.01224130775263959
0.008253303376695118
accuracy rate = 0.9
"time: 132364042 micro second"
"-------------"
:ok
