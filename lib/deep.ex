

defmodule Test do

  def init_network1() do
    [Dmatrix.new(784,50,0.1),
     Matrix.new(1,50),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     4.3,
     Dmatrix.new(50,100,0.1),
     Matrix.new(1,100),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     4.2,
     Dmatrix.new(100,10,0.1),
     Matrix.new(1,10),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     4.1]
  end

  def init_network2() do
    [Dmatrix.new(12,6,0.1),
     Matrix.new(1,6),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     1,
     Dmatrix.new(6,2,0.1),
     Matrix.new(1,2),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     1]
  end

  def dt() do
    [[[1,1,1,
       1,0,1,
       1,0,1,
       1,1,1]],
      [[1,0]],
      [[1,0,1,
        0,1,0,
        0,1,1,
        1,0,1]],
       [[0,1]],
      [[1,0,1,
        0,1,0,
        0,1,1,
        1,0,1]],
       [[0,1]],
     [[1,1,1,
       1,0,1,
       1,0,1,
       1,1,0]],
     [[1,0]],
     [[1,1,1,
       1,0,1,
       1,0,1,
       0,1,1]],
     [[1,0]],
     [[0,0,0,
       1,1,1,
       1,0,1,
       1,1,1]],
     [[1,0]],
     [[0,0,0,
       0,1,1,
       1,0,1,
       1,1,1]],
     [[1,0]],
     [[1,0,1,
       0,1,0,
       1,0,1,
       0,0,0]],
      [[0,1]],
     [[1,0,1,
       0,1,0,
       1,0,1,
       1,0,0]],
      [[0,1]],
     [[1,0,1,
       0,1,0,
       1,0,1,
       0,0,1]],
      [[0,1]],
      [[0,1,1,
        1,0,1,
        1,0,1,
        1,1,1]],
       [[1,0]],
      [[1,1,0,
        1,0,1,
        1,0,1,
        1,1,0]],
      [[1,0]],
     [[1,0,1,
       0,1,0,
       0,1,0,
       1,0,1,]],
      [[0,1]],
     [[1,0,1,
       0,1,0,
       1,1,0,
       1,0,1]],
      [[0,1]],
     [[1,0,1,
       1,1,0,
       0,1,0,
       1,0,1]],
      [[0,1]]]
  end
end


defmodule DL do
  def sigmoid(x) do
    cond do
      x > 100 -> 1
      x < -100 -> 0
      true ->  1 / (1+:math.exp(-x))
    end
  end

  def dsigmoid(x) do
    (1 - sigmoid(x)) * sigmoid(x)
  end

  def step(x) do
    if x > 0 do 1 else 0 end
  end

  def relu(x) do
    max(0,x)
  end

  def drelu(x) do
    if x > 0 do 1 else 0 end
  end

  def ident(x) do
    x
  end

  def dident(_) do
    1
  end

  def softmax(x) do
    sum = Enum.reduce(x, fn(y, acc) -> :math.exp(y) + acc end)
    Enum.map(x, fn(y) -> :math.exp(y)/sum end)
  end


  #error function
  def cross_entropy([x],[y]) do
    cross_entropy1(x,y)
  end
  def cross_entropy1([],[]) do 0 end
  def cross_entropy1([y|ys],[t|ts]) do
    delta = 1.0e-7
    -(t * :math.log(y+delta)) + cross_entropy1(ys,ts)
  end

  def mean_square([x],[t]) do
    mean_square1(x,t) / 2
  end

  def mean_square1([],[]) do 0 end
  def mean_square1([x|xs],[t|ts]) do
    square(x-t) + mean_square1(xs,ts)
  end

  def square(x) do
    x*x
  end

  # apply functin for matrix
  def apply_function([],_) do [] end
  def apply_function([x|xs],f) do
    [Enum.map(x,fn(y) -> f.(y) end)|apply_function(xs,f)]
  end

  def forward([],x) do x end
  def forward([w,b,f,_,_|rest],x) do
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b) |> apply_function(f)
    forward(rest,x1)
  end

  # for backpropagation
  def forward_for_back(network,x) do
    forward_for_back1(network,x,[x])
  end
  def forward_for_back1([],_,res) do res end
  def forward_for_back1([w,b,f,_,_|rest],x,res) do
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b)
    x2 = x1 |> apply_function(f)
    forward_for_back1(rest,x2,[x2,x1|res])
  end

  # for numerical gradient
  def forward_w([],x,_,_,_,_) do x end
  def forward_w([w,b,f,_,_|rest],x,0,r,c,d) do
    w1 = Dmatrix.diff(w,r,c,d)
    x1 = Dmatrix.mult(x,w1)|> Dmatrix.add(b) |> apply_function(f)
    forward_w(rest,x1,-1,r,c,d)
  end
  def forward_w([w,b,f,_,_|rest],x,n,r,c,d) do
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b) |> apply_function(f)
    forward_w(rest,x1,n-1,r,c,d)
  end

  # for numerical gradient
  def forward_b([],x,_,_,_) do x end
  def forward_b([w,b,f,_,_|rest],x,0,c,d) do
    b1 = Dmatrix.diff(b,0,c,d)
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b1) |> apply_function(f)
    forward_b(rest,x1,-1,c,d)
  end
  def forward_b([w,b,f,_,_|rest],x,n,c,d) do
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b) |> apply_function(f)
    forward_b(rest,x1,n-1,c,d)
  end

  # numerical gradient
  def numerical_gradient(network,x,t) do
    numerical_gradient1(network,x,t,0,network)
  end

  def numerical_gradient1([],_,_,_,_) do [] end
  def numerical_gradient1([w,b,f,g,r|rest],x,t,n,network) do
    [numerical_gradient_w(w,x,n,network,t),numerical_gradient_b(b,x,n,network,t),f,g,r|
     numerical_gradient1(rest,x,t,n+1,network)]
  end

  # numerical gradient for wight matrix
  def numerical_gradient_w(w,x,n,network,t) do
    {r,c} = Matrix.size(w)
    Enum.map(0..r-1,
      fn(x1) -> Enum.map(0..c-1,
                  fn(y1) -> numerical_gradient_w1(x,x1,y1,n,network,t) end) end)
  end

  def numerical_gradient_w1(x,r,c,n,network,t) do
    h = 0.0001
    y0 = forward(network,x)
    y1 = forward_w(network,x,n,r,c,h)
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end

  # numerical gradient bias row vector
  def numerical_gradient_b(b,x,n,network,t) do
    {_,c} = Matrix.size(b)
    [Enum.map(0..c-1,fn(y1) -> numerical_gradient_b1(x,y1,n,network,t) end)]
  end

  def numerical_gradient_b1(x,c,n,network,t) do
    h = 0.0001
    y0 = forward(network,x)
    y1 = forward_b(network,x,n,c,h)
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end

  # gradient with backpropagation
  def gradient(network,x,t) do
    x1 = forward_for_back(network,x)
    l = Matrix.sub(hd(x1),t)
    backpropagation(network,l,tl(x1))
  end

  def backpropagation(network,l,u) do
    backpropagation1(Enum.reverse(network),l,u,[])
  end

  def backpropagation1([],_,_,res) do res end
  def backpropagation1([r,g,f,_,w|rest],l,[u1,u2|us],res) do
    l1 = Matrix.emult(l,DL.apply_function(u1,g))
    l2 = Dmatrix.mult(l1,Matrix.transpose(w))
    w1 = Dmatrix.mult(Matrix.transpose(u2),l1)
    backpropagation1(rest,l2,us,[w1,l1,f,g,r|res])
  end

  # update wight and bias
  def learning([],_) do [] end
  def learning([w,b,f,g,r|rest],[w1,b1,_,_,_|gradrest]) do
    [Dmatrix.update(w,w1,r),Dmatrix.update(b,b1,r),f,g,r|
     learning(rest,gradrest)]
  end

  # stochastic gradient descent
  def sgd() do
    network = Test.init_network2()
    dt = Test.dt()
    network1 = sgd1(network,dt,100)
    print(forward(network1,Enum.at(dt,0)))
    print(Enum.at(dt,1))
    newline()
    print(forward(network1,Enum.at(dt,2)))
    print(Enum.at(dt,3))
    newline()
    dt = [[1,1,1,
           0,0,1,
           1,0,1,
           1,1,1]]
    print(forward(network1,dt))
  end

  def sgd1(network,_,0) do  network end
  def sgd1(network,dt,n) do
    network1 = sgd2(network,dt,0)
    sgd1(network1,dt,n-1)
  end

  def sgd2(network,[],error) do
    IO.puts(error)
    network
  end
  def sgd2(network,[x,t|rest],error) do
    network1 = gradient(network,x,t)
    network2 = learning(network,network1)
    x1 = forward(network,x)
    error1 = mean_square(x1,t)
    sgd2(network2,rest,error1+error)
  end


  def print(x) do
    :io.write(x)
  end

  def newline() do
    IO.puts("")
  end

  def print_network([]) do
    IO.puts("")
  end
  def print_network([x|xs]) do
    if is_list(x) do
      Dmatrix.print(x)
    else
      :io.write(x)
    end
    print_network(xs)
  end

end

#DL for mini_batch
defmodule DLB do
  def print(x) do
    :io.write(x)
  end
  # x is matrix. This is batch data
  def forward([],x) do x end
  def forward([w,b,f,_,_|rest],x) do
    {e,_} = Matrix.size(x)
    b1 = Dmatrix.expand(b,e)
    x1 = Pmatrix.mult(x,w)|> Matrix.add(b1) |> DL.apply_function(f)
    forward(rest,x1)
  end

  # for backpropagation
  def forward_for_back(network,x) do
    forward_for_back1(network,x,[x])
  end
  def forward_for_back1([],_,res) do res end
  def forward_for_back1([w,b,f,_,_|rest],x,res) do
    {e,_} = Matrix.size(x)
    b1 = Dmatrix.expand(b,e)
    x1 = Pmatrix.mult(x,w)|> Dmatrix.add(b1)
    x2 = x1 |> DL.apply_function(f)
    forward_for_back1(rest,x2,[x2,x1|res])
  end

  # for numerical gradient
  def forward_w([],x,_,_,_,_) do x end
  def forward_w([w,b,f,_,_|rest],x,0,r,c,d) do
    w1 = Dmatrix.diff(w,r,c,d)
    {e,_} = Matrix.size(x)
    b1 = Dmatrix.expand(b,e)
    x1 = Dmatrix.mult(x,w1)|> Dmatrix.add(b1) |> DL.apply_function(f)
    forward_w(rest,x1,-1,r,c,d)
  end
  def forward_w([w,b,f,_,_|rest],x,n,r,c,d) do
    {e,_} = Matrix.size(x)
    b1 = Dmatrix.expand(b,e)
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b1) |> DL.apply_function(f)
    forward_w(rest,x1,n-1,r,c,d)
  end

  # for numerical gradient
  def forward_b([],x,_,_,_) do x end
  def forward_b([w,b,f,_,_|rest],x,0,c,d) do
    b1 = Dmatrix.diff(b,0,c,d)
    {e,_} = Matrix.size(x)
    b2 = Dmatrix.expand(b1,e)
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b2) |> DL.apply_function(f)
    forward_b(rest,x1,-1,c,d)
  end
  def forward_b([w,b,f,_,_|rest],x,n,c,d) do
    {e,_} = Matrix.size(x)
    b1 = Dmatrix.expand(b,e)
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b1) |> DL.apply_function(f)
    forward_b(rest,x1,n-1,c,d)
  end
  # numerical gradient
  def numerical_gradient(network,x,t) do
    numerical_gradient1(network,x,t,0,network)
  end

  def numerical_gradient1([],_,_,_,_) do [] end
  def numerical_gradient1([w,b,f,g,r|rest],x,t,n,network) do
    [numerical_gradient_w(w,x,n,network,t),numerical_gradient_b(b,x,n,network,t),f,g,r|
     numerical_gradient1(rest,x,t,n+1,network)]
  end

  # numerical gradient for wight matrix
  def numerical_gradient_w(w,x,n,network,t) do
    {r,c} = Matrix.size(w)
    Enum.map(0..r-1,
      fn(x1) -> Enum.map(0..c-1,
                  fn(y1) -> numerical_gradient_w1(x,x1,y1,n,network,t) end) end)
  end

  def numerical_gradient_w1(x,r,c,n,network,t) do
    h = 0.0001
    y0 = forward(network,x)
    y1 = forward_w(network,x,n,r,c,h)
    (batch_error(y1,t) - batch_error(y0,t)) / h
  end

  def batch_error(y,t) do
    batch_error1(y,t,0) / length(y)
  end

  def batch_error1([],[],res) do res end
  def batch_error1([y|ys],[t|ts],res) do
    batch_error1(ys,ts,DL.mean_square([t],[y])+res)
  end

  # numerical gradient bias row vector
  def numerical_gradient_b(b,x,n,network,t) do
    {_,c} = Matrix.size(b)
    [Enum.map(0..c-1,fn(y1) -> numerical_gradient_b1(x,y1,n,network,t) end)]
  end

  def numerical_gradient_b1(x,c,n,network,t) do
    h = 0.0001
    y0 = forward(network,x)
    y1 = forward_b(network,x,n,c,h)
    (batch_error(y1,t) - batch_error(y0,t)) / h
  end

  # gradient with backpropagation x and t are matrix
  def gradient(network,x,t) do
    x1 = forward_for_back(network,x)
    l = Matrix.sub(hd(x1),t)
    backpropagation(network,l,tl(x1))
  end

  def backpropagation(network,l,u) do
    backpropagation1(Enum.reverse(network),l,u,[])
  end

  def backpropagation1([],_,_,res) do res end
  def backpropagation1([r,g,f,_,w|rest],l,[u1,u2|us],res) do
    {n,_} = Matrix.size(l)
    l1 = Matrix.emult(l,DL.apply_function(u1,g))
    l2 = Pmatrix.mult(l1,Matrix.transpose(w))
    w1 = Pmatrix.mult(Matrix.transpose(u2),l1) |> DL.apply_function(fn(x) -> x/n end)
    b1 = Dmatrix.reduce(l1) |> DL.apply_function(fn(x) -> x/n end) #bias
    backpropagation1(rest,l2,us,[w1,b1,f,g,r|res])
  end

  def mnist(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image()
    label = MNIST.train_label()
    network = Test.init_network1()
    test_image = MNIST.test_image()
    test_label = MNIST.test_label()
    IO.puts("c error")
    network1 = batch(network,image,label,m,n)
    IO.puts("verifying")
    c = accuracy(network1,test_image,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(c/100)
  end

  def batch(network,_,_,_,0) do network end
  def batch(network,image,train,m,n) do
    IO.write("rest epoch = ")
    IO.puts(n)
    network1 = mini_batch(network,image,train,m)
    batch(network1,image,train,m,n-1)
  end

  def mini_batch(network,_,_,0) do network end
  def mini_batch(network,image,train,m) do
    mini_image = Enum.map(Enum.take(image,10),fn(y) -> MNIST.normalize(y,255) end)
    mini_train = Enum.map(Enum.take(train,10),fn(y) -> MNIST.to_onehot(y) end)
    network1 = gradient(network,mini_image,mini_train)
    network2 = DL.learning(network,network1)
    error = batch_error(network2,mini_image,mini_train)
    IO.write("mini batch error = ")
    IO.puts(error)
    mini_batch(network2,Enum.drop(image,10),Enum.drop(train,10),m-10)
  end

  def batch_error(network,image,train) do
    y = forward(network,image)
    Matrix.sub(y,train) |> DL.apply_function(fn(y) -> DL.square(y) end) |> Dmatrix.sum
  end

  # print predict of test data
  def accuracy(_,_,_,0,correct) do
    correct
  end
  def accuracy(network,[image|irest],[label|lrest],n,correct) do
    dt = MNIST.onehot_to_num(forward(network,[MNIST.normalize(image,255)]))
    if dt != label do
      accuracy(network,irest,lrest,n-1,correct)
    else
      accuracy(network,irest,lrest,n-1,correct+1)
    end
  end

end
