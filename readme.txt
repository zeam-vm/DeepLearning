For my study Deep Learning with elixir

Usage:
  iex -S mix
  DLB.mnist(size,epoch) # test with MNIST data set

Example:
iex(1)> DLB.mnist(1000,30)
prepareing data
...
mini batch error = 1.0368650214598516
mini batch error = 0.9950205959254711
mini batch error = 0.026115933479561745
mini batch error = 0.0395832162443519
verifying
accuracy rate = 0.89
:ok
iex(2)>

module DL is basic code for Deep learning
module DLB is for batch processing
module Dmatrix is code for Matrix
module Pmatrix is code for Matrix product in paralell
module MNIST is code for MNIST data set

I implemented backpropagation and numerical-gradient
Now I'm testing small data set.

I implemented CNN partialy.
