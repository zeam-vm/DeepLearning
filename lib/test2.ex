defmodule Btest do
  import BLASNetwork
  defnetwork init_network1(_x) do
    _x |> w(2,2) |> b(2) |> softmax
  end

  defnetwork init_network2(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300,0.5) |> b(300,0.5) |> sigmoid
    |> w(300,100,0.5) |> b(100,0.5) |> sigmoid
    |> w(100,10,0.5) |> b(10,0.5) |> softmax
  end


  def test1() do
    network = init_network1(0)
    dt = Matrex.new([[1,2]])
    BLASDP.forward(dt,network)
  end

  def test2() do
    IO.puts("preparing data")
    network = init_network2(0)
    image = MNIST.train_image(10) |> Ctensor.to_matrex
    train = MNIST.train_label_onehot(10) |> Cmatrix.to_matrex
    BLASDPB.numerical_gradient(image,network,train)
  end
end
