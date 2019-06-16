defmodule Test do
  import Network

  # for sgd test
  defnetwork init_network2(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300) |> b(300) |> sigmoid
    |> w(300,100) |> b(100) |> sigmoid
    |> w(100,10) |> b(10) |> softmax
  end

  # for momentum test
  defnetwork init_network3(_x) do
    _x |> f(5,5,0.03) |> flatten
    |> w(576,300,0.03) |> b(300,0.03) |> relu
    |> w(300,100,0.03) |> b(100,0.03) |> relu
    |> w(100,10,0.03) |> b(10,0.03) |> softmax
  end

  # for adagrad test
  defnetwork init_network4(_x) do
    _x |> f(5,5,0.03) |> flatten
    |> w(576,300,0.03) |> b(300,0.03) |> relu
    |> w(300,100,0.01) |> b(100,0.01) |> relu
    |> w(100,10,0.005) |> b(10,0.005) |> softmax
  end

  # for adam test
  defnetwork init_network5(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300) |> b(300) |> relu
    |> w(300,100) |> b(100) |> relu
    |> w(100,10) |> b(10) |> softmax
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
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def sgd1(_,network,_,_,0) do network end
  def sgd1(image,network,train,m,n) do
    {image1,train1} = DP.random_select(image,train,m,2000)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1)
    y = DP.forward(image1,network2)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    sgd1(image,network2,train,m,n-1)
  end

  def momentum(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000) |> Ctensor.to_matrex
    label = MNIST.train_label_onehot(3000)
    network = init_network3(0)
    test_image = MNIST.test_image(1000) |> Ctensor.to_matrex
    test_label = MNIST.test_label(1000)
    IO.puts("ready")
    network1 = momentum1(image,network,label,m,n)
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def momentum1(_,network,_,_,0) do network end
  def momentum1(image,network,train,m,n) do
    {image1,train1} = DP.random_select(image,train,m,2000)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1,:momentum)
    y = DP.forward(image1,network2)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    momentum1(image,network2,train,m,n-1)
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
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def adagrad1(_,network,_,_,0) do network end
  def adagrad1(image,network,train,m,n) do
    {image1,train1} = DP.random_select(image,train,m,2000)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1,:adagrad)
    y = DP.forward(image1,network2)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    adagrad1(image,network2,train,m,n-1)
  end

  def adam(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000) |> Ctensor.to_matrex
    label = MNIST.train_label_onehot(3000)
    network = init_network4(0)
    test_image = MNIST.test_image(1000) |> Ctensor.to_matrex
    test_label = MNIST.test_label(1000)
    IO.puts("ready")
    network1 = adam1(image,network,label,m,n)
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def adam1(_,network,_,_,0) do network end
  def adam1(image,network,train,m,n) do
    {image1,train1} = DP.random_select(image,train,m,2000)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1,:adam)
    y = DP.forward(image1,network2)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    adam1(image,network2,train,m,n-1)
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
    network1 = DP.numerical_gradient(dt,network,tt,:cross)
    network2 = DP.gradient(dt,network,tt)
    IO.inspect(network1)
    IO.inspect(network2)
    true
  end

end
