defmodule Test do
  import Network
  defnetwork init_network1(_x) do
    _x |> w(784,50) |> b(50) |> sigmoid
    |> w(50,100) |> b(100) |> sigmoid
    |> w(100,10) |> b(10) |> sigmoid
  end

  # for sgd test
  defnetwork init_network2(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300,0.5) |> b(300,0.5) |> sigmoid
    |> w(300,100,0.5) |> b(100,0.5) |> sigmoid
    |> w(100,10,0.5) |> b(10,0.5) |> softmax
  end

  # for momentum test
  defnetwork init_network3(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300,0.05) |> b(300,0.05) |> relu
    |> w(300,100,0.05) |> b(100,0.05) |> relu
    |> w(100,10,0.05) |> b(10,0.05) |> softmax
  end

  # for adagrad test
  defnetwork init_network4(_x) do
    _x |> f(5,5,0.02) |> flatten
    |> w(576,300,0.02) |> b(300,0.02) |> relu
    |> w(300,100,0.02) |> b(100,0.02) |> relu
    |> w(100,10,0.02) |> b(10,0.02) |> softmax
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
    image = MNIST.train_image(2000)
    label = MNIST.train_label_onehot(2000)
    network = init_network2(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = sgd1(image,network,label,m,n)
    correct = DPB.accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def sgd1(_,network,_,_,0) do network end
  def sgd1(image,network,train,m,n) do
    {image1,train1} = DPB.random_select(image,train,[],[],m)
    network1 = DPP.gradient(image1,network,train1)
    network2 = DPB.learning(network,network1)
    y = DPB.forward(image1,network2)
    loss = DPB.batch_error(y,train1,fn(x,y) -> DP.cross_entropy(x,y) end)
    DP.print(loss)
    DP.newline()
    sgd1(image,network2,train,m,n-1)
  end

  def momentum(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(2000)
    label = MNIST.train_label_onehot(2000)
    network = init_network3(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = momentum1(image,network,label,m,n)
    correct = DPB.accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def momentum1(_,network,_,_,0) do network end
  def momentum1(image,network,train,m,n) do
    {image1,train1} = DPB.random_select(image,train,[],[],m)
    network1 = DPP.gradient(image1,network,train1)
    network2 = DPB.learning(network,network1,:momentum)
    y = DPB.forward(image1,network2)
    loss = DPB.batch_error(y,train1,fn(x,y) -> DP.cross_entropy(x,y) end)
    DP.print(loss)
    DP.newline()
    momentum1(image,network2,train,m,n-1)
  end

  def adagrad(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(2000)
    label = MNIST.train_label_onehot(2000)
    network = init_network4(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = adagrad1(image,network,label,m,n)
    correct = DPB.accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def adagrad1(_,network,_,_,0) do network end
  def adagrad1(image,network,train,m,n) do
    {image1,train1} = DPB.random_select(image,train,[],[],m)
    network1 = DPP.gradient(image1,network,train1)
    network2 = DPB.learning(network,network1,:adagrad)
    y = DPB.forward(image1,network2)
    loss = DPB.batch_error(y,train1,fn(x,y) -> DP.cross_entropy(x,y) end)
    DP.print(loss)
    DP.newline()
    adagrad1(image,network2,train,m,n-1)
  end

  def adam(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(2000)
    label = MNIST.train_label_onehot(2000)
    network = init_network4(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = adam1(image,network,label,m,n)
    correct = DPB.accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def adam1(_,network,_,_,0) do network end
  def adam1(image,network,train,m,n) do
    {image1,train1} = DPB.random_select(image,train,[],[],m)
    network1 = DPP.gradient(image1,network,train1)
    network2 = DPB.learning(network,network1,:adam)
    y = DPB.forward(image1,network2)
    loss = DPB.batch_error(y,train1,fn(x,y) -> DP.cross_entropy(x,y) end)
    DP.print(loss)
    DP.newline()
    adam1(image,network2,train,m,n-1)
  end

  # MNIST test
  def batch(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(m)
    label = MNIST.train_label_onehot(m)
    network = init_network2(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = batch1(image,network,label,n)
    correct = DPB.accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def batch1(_,network,_,0) do network end
  def batch1(image,network,train,n) do
    network1 = DPB.gradient(image,network,train)
    network2 = DPB.learning(network,network1)
    y = DPB.forward(image,network1)
    loss = DPB.batch_error(y,train,fn(x,y) -> DP.mean_square(x,y) end)
    DP.print(loss)
    DP.newline()
    batch1(image,network2,train,n-1)
  end

  #online
  def online(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(m)
    label = MNIST.train_label(m)
    network = init_network2(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = online1(image,network,label,m,n)
    correct = DP.accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def online1(_,network,_,_,0) do network end
  def online1(image,network,label,m,n) do
    network1= online2(image,network,label,m)
    train1 = MNIST.to_onehot(hd(label))
    y = DP.forward(hd(image),network1)
    loss = DP.mean_square(y,train1)
    DP.print(loss)
    DP.newline()
    online1(image,network1,label,m,n-1)
  end

  def online2(_,network,_,0) do network end
  def online2([image1|image],network,[label1|label],m) do
    train1 = MNIST.to_onehot(label1)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1)
    online2(image,network2,label,m-1)
  end

  defnetwork check_network(_x) do
    _x |> cw([[0.1,0.2,0.3,0.4],
              [0.3,0.2,0.1,0.4],
              [0.2,0.2,0.2,0.2],
              [0.4,0.2,0.3,0.1]]) |> b(2)
    |> softmax
  end
  def check() do
    dt = [[0.1,0.2,0.03,0.04],[0.2,0.03,0.04,0.05]]
    tt = [[0,1],[1,0]]
    network = check_network(0)
    #res = DPB.forward(dt,network)
    #DP.print(res)
    #DPB.batch_error(res,tt,fn(x,y) -> DP.cross_entropy(x,y) end)
    network1 = DPB.numerical_gradient(dt,network,tt,:cross)
    network2 = DPB.gradient(dt,network,tt)
    DP.print(network1)
    DP.newline()
    DP.print(network2)
  end

end
