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

I implemented CNN and testing.
New frame work is called Function Flow (FF)
module FF
This module is extremely incomplete.Now I am improving

expample:
iex(3)> FF.sgd(1000,10)
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
