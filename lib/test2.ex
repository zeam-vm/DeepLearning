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
    #image = Cmatrix.to_matrex([[1,2],[3,4]])
    #train = Cmatrix.to_matrex([[0,1],[1,2]])
    image = MNIST.train_image(2) |> Ctensor.to_matrex
    train = MNIST.train_label_onehot(2) |> Cmatrix.to_matrex
    #IO.inspect(BLASDPB.numerical_gradient(image,network,train))
    IO.inspect(BLASDPB.gradient(image,network,train))
  end

  def sgd(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000) |> Ctensor.to_matrex
    label = MNIST.train_label_onehot(3000)
    network = init_network2(0)
    test_image = MNIST.test_image(1000) |> Ctensor.to_matrex
    test_label = MNIST.test_label(1000)
    IO.puts("ready")
    network1 = sgd1(image,network,label,m,n)
    correct = BLASDPB.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def sgd1(_,network,_,_,0) do network end
  def sgd1(image,network,train,m,n) do
    {image1,train1} = BLASDPB.random_select(image,train,m,2000)
    network1 = BLASDPB.gradient(image1,network,train1)
    network2 = BLASDPB.learning(network,network1)
    y = BLASDPB.forward(image1,network2)
    loss = BLASDPB.loss(y,train1,:square)
    DP.print(loss)
    DP.newline()
    sgd1(image,network2,train,m,n-1)
  end

end
