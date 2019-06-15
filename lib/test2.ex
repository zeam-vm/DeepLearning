defmodule Btest do
  import BLASNetwork
  defnetwork init_network1(_x) do
    _x |> cw([[0.1,0.2],[0.3,0.4]]) |> b(2) |> softmax
  end

  # for sgd
  defnetwork init_network2(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,10) |> b(10) |> softmax
  end

  # for adagrad test
  defnetwork init_network4(_x) do
    _x |> f(5,5,0.04) |> flatten
    |> w(576,100,0.04) |> b(100,0.04) |> relu
    |> w(100,10,0.01) |> b(10,0.01) |> softmax
  end


  def test1() do
    network = init_network1(0)
    dt = Matrex.new([[1,2]])
    BLASDP.forward(dt,network)
  end

  def test2() do
    IO.puts("preparing data")
    network = init_network1(0)
    image = Cmatrix.to_matrex([[1,2],[3,4]])
    train = Cmatrix.to_matrex([[0,1],[1,2]])
    #BLASDPB.forward(image,network)
    #BLASDPB.forward_for_back(image,network,[])
    #image = MNIST.train_image(2) |> Ctensor.to_matrex
    #|> Enum.each(fn(x) -> Matrex.heatmap(x) end)
    #train = MNIST.train_label_onehot(2) |> Cmatrix.to_matrex
    #IO.inspect(train)
    IO.inspect(BLASDPB.numerical_gradient(image,network,train,:cross))
    IO.inspect(BLASDPB.gradient(image,network,train))
    true
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

  def adagrad(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000) |> Ctensor.to_matrex
    label = MNIST.train_label_onehot(3000)
    network = init_network4(0)
    test_image = MNIST.test_image(1000) |> Ctensor.to_matrex
    test_label = MNIST.test_label(1000)
    IO.puts("ready")
    network1 = adagrad1(image,network,label,m,n)
    correct = BLASDPB.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def adagrad1(_,network,_,_,0) do network end
  def adagrad1(image,network,train,m,n) do
    {image1,train1} = BLASDPB.random_select(image,train,m,2000)
    network1 = BLASDPB.gradient(image1,network,train1)
    network2 = BLASDPB.learning(network,network1,:adagrad)
    y = BLASDPB.forward(image1,network2)
    loss = BLASDPB.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    adagrad1(image,network2,train,m,n-1)
  end

  defnetwork check_network(_x) do
    _x |> cf([[0.1,0.2],[0.3,0.4]]) |> flatten
    |> cw([[0.1,0.2],
              [0.3,0.2],
              [0.2,0.2],
              [0.4,0.2]]) |> cb([[0,0]]) |> sigmoid
    |> cw([[0.1,0.2],[0.2,0.1]]) |> cb([[0,0]]) |> softmax
  end
  def check() do
    dt = Ctensor.to_matrex([[[0.1,0.2,0.03],
                             [0.04,0.5,0.3],
                             [0.2,0.03,0.04]],
                            [[0.2,0.3,0.2],
                             [0.1,0.2,0.1],
                             [0.3,0.3,0.2]],
                            [[0.2,0.1,0.1],
                             [0.2,0.1,0.4],
                             [0.2,0.1,0.3]]])
    tt = Cmatrix.to_matrex([[0,1],[1,0],[1,0]])
    network = check_network(0)
    #res = BLASDPB.forward(dt,network)
    #DP.print(res)
    #DPB.batch_error(res,tt,fn(x,y) -> DP.cross_entropy(x,y) end)
    network1 = BLASDPB.numerical_gradient(dt,network,tt,:cross)
    network2 = BLASDPB.gradient(dt,network,tt)
    IO.inspect(network1)
    IO.inspect(network2)
    true
  end

end
