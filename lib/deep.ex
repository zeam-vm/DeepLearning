defmodule Test do
  def init_network1() do
    [Dmatrix.new(784,50,0.1),
     Matrix.new(1,50),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     3.2,
     Dmatrix.new(50,100,0.1),
     Matrix.new(1,100),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     3.1,
     Dmatrix.new(100,10,0.1),
     Matrix.new(1,10),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     3]
  end

  def init_network2() do
    [Dmatrix.new(12,8,1),
     [[0,0,0,0,0,0,0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     5,
     Dmatrix.new(8,6,1),
     [[0,0,0,0,0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     5,
     Dmatrix.new(6,3,1),
     [[0,0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     5,
     Dmatrix.new(3,2,1),
     [[0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     5]
  end

  def init_network3() do
    [[[1,2,3,4,5,6,7,8],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9],
      [2,3,4,5,6,7,8,9]],
     [[0,0,0,0,0,0,0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     1,
     [[1,2,3,4,5,6],
      [1,2,3,4,5,6],
      [1,2,3,4,5,6],
      [1,2,3,4,5,6],
      [1,2,3,4,5,6],
      [1,2,3,4,5,6],
      [1,2,3,4,5,6],
      [1,2,3,4,5,6]],
     [[0,0,0,0,0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     1,
     [[1,2,3],
      [1,2,3],
      [1,2,3],
      [1,2,3],
      [1,2,3],
      [1,2,3]],
     [[0,0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     1,
     [[1,2],
      [1,2],
      [1,2]],
     [[0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     1]
  end


  def dt1 do
    [[1,1,1,1,1,1,1,1,1,1,1,1]]
  end
  def dt2 do
    [[1,1,1,1,1,0,1,1,1,1,1,0]]
  end
  def dt3 do
    [[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,0,1,1,1,1,1,0]]
  end
  def tt3 do
    [[1,0],[0,1]]
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
    [Dmatrix.element_mult(w,w1,r),Dmatrix.element_mult(b,b1,r),f,g,r|
     learning(rest,gradrest)]
  end

  # stochastic gradient descent
  def sgd() do
    network = Test.init_network1()
    dt = Test.dt()
    network1 = mini_batch(network,dt,100)
    print(forward(network1,Enum.at(dt,0)))
    print(Enum.at(dt,1))
    print(forward(network1,Enum.at(dt,2)))
    print(Enum.at(dt,3))
    dt = [[1,1,1,
           0,0,1,
           1,0,1,
           1,1,1]]
    print(forward(network1,dt))
  end

  def mini_batch(network,_,0) do  network end
  def mini_batch(network,dt,n) do
    network1 = mini_batch1(network,dt,0)
    mini_batch(network1,dt,n-1)
  end

  def mini_batch1(network,[],error) do
    IO.puts(error)
    network
  end
  def mini_batch1(network,[x,t|rest],error) do
    network1 = gradient(network,x,t)
    network2 = learning(network,network1)
    x1 = forward(network,x)
    error1 = mean_square(x1,t)
    mini_batch1(network2,rest,error1+error)
  end

  def mnist(m,n) do
    IO.puts("prepareing data")
    image = MNIST.train_image()
    label = MNIST.train_label()
    network = Test.init_network1()
    seq = rand_sequence(m,length(image))
    test_image = MNIST.test_image()
    test_label = MNIST.test_label()
    IO.puts("c error")
    network1 = batch(network,image,label,m,n,seq)
    IO.puts("verifying")
    c = accuracy(network1,test_image,test_label,10000,0)
    IO.write("accuracy rate = ")
    IO.puts(c/10000)
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

  def batch(network,_,_,_,0,_) do network end
  def batch(network,_,_,_,_,[]) do network end
  def batch(network,image,label,n,c,seq) do
    {network1,error} = batch1(network,image,label,0,seq,0)
    print(c)
    IO.write(" ")
    print(error)
    newline()
    batch(network1,image,label,n,c-1,seq)
  end

  def batch1(network,_,_,60000,_,error) do
    {network,error}
  end
  def batch1(network,_,_,_,[],error) do
    {network,error}
  end
  def batch1(network,[image|irest],[label|lrest],n,[n|srest],error) do
    x = [MNIST.normalize(image,255)]
    t = [MNIST.to_onehot(label)]
    network1 = gradient(network,x,t)
    network2 = learning(network,network1)
    x1 = forward(network2,x)
    error1 = mean_square(x1,t)
    batch1(network2,irest,lrest,n+1,srest,error1+error)
  end
  def batch1(network,[_|irest],[_|lrest],n,[seq|srest],error) do
    batch1(network,irest,lrest,n+1,[seq|srest],error)
  end

  #for mini batch
  def rand_sequence(c,n) do
    rand_sequence1(c,n,[]) |> Enum.sort
  end

  def rand_sequence1(0,_,res) do res end
  def rand_sequence1(c,n,res) do
    rand_sequence1(c-1,n,[:rand.uniform(n)|res])
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
    l3 = Dmatrix.reduce(l1) |> DL.apply_function(fn(x) -> x/n end) #bias
    backpropagation1(rest,l2,us,[w1,l3,f,g,r|res])
  end

  def mnist(m,n) do
    IO.puts("prepareing data")
    image = MNIST.train_image()
    label = MNIST.train_label()
    network = Test.init_network1()
    test_image = MNIST.test_image()
    test_label = MNIST.test_label()
    IO.puts("c error")
    network1 = batch(network,image,label,m,n)
    IO.puts("verifying")
    c = DL.accuracy(network1,test_image,test_label,100,0)
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


end


defmodule Dmatrix do
  # box-muller rand
  def box_muller() do
    x = :rand.uniform()
    y = :rand.uniform()
    :math.sqrt(-2.0 * :math.log(x)) * :math.cos(2.0 * :math.pi * y);
  end

  #generate initial wweight matrix with box-muller
  def new(0,_,_) do [] end
  def new(r,c,rate) do
    [new1(c,rate)|new(r-1,c,rate)]
  end

  def new1(0,_) do [] end
  def new1(c,rate) do
    [box_muller()*rate|new1(c-1,rate)]
  end

  def print([]) do
    IO.puts("")
  end
  def print([x|xs]) do
    :io.write(x)
    IO.puts("")
    print(xs)
  end

  def mult(x,y) do
    {r0,c} = Matrix.size(x)
    {r,_} = Matrix.size(y)
    if r != c do
      IO.puts("Dmatrix mult error")
      :io.write(x)
      :io.write(y)
    else
      if r0 == 1 and c >= 100 do
        pmult(x,y)
      else
        Matrix.mult(x,y)
      end
    end
  end

  def pmult(x,y) do
    y1 = Matrix.transpose(y)
    {r,c} = Matrix.size(x)
    {r1,c1} = Matrix.size(y)
    d = 2
    if c != r1 do
      :error
    else if r > 1 or c < 50 do
            Matrix.mult(x,y)
         else
            pmult1(x,y1,c1,c1,lot(c1,d),last_lot(c1,d))
            ans = pmult2(d,[])
            |> Enum.sort
            |> Enum.map(fn(x) -> Enum.drop(x,1) |> hd end)
            |> flatten
            [ans]
        end
    end
  end

  def flatten([]) do [] end
  def flatten([x|xs]) do
    x ++ flatten(xs)
  end

  def lot(m,c) do
    div(m,c)
  end

  def last_lot(m,c) do
    div(m,c) + rem(m,c)
  end

  def pmult1(_,_,_,0,_,_) do true end
  def pmult1(x,y,m,m,l1,l2) do
    pid = spawn(Worker,:part,[])
    send pid, {self(),{m,x,Enum.slice(y,m-l2,l2)}}
    pmult1(x,y,m,m-l2,l1,l2)
  end
  def pmult1(x,y,m,c,l1,l2) do
    pid = spawn(Worker,:part,[])
    send pid, {self(),{c,x,Enum.slice(y,c-l1,l1)}}
    pmult1(x,y,m,c-l1,l1,l2)
  end


  def pmult2(0,res) do res end
  def pmult2(d,res) do
    receive do
      {:answer,ls} ->
        pmult2(d-1,[ls|res])
    end
  end

  def add(x,y) do
    {r1,c1} = Matrix.size(x)
    {r2,c2} = Matrix.size(y)
    if r1 != r2 or c1 != c2 do
      IO.puts("Dmatrix add error")
      :io.write(x)
      :io.write(y)
    else
      Matrix.add(x,y)
    end
  end

  # for learning
  def element_mult([],[],_) do [] end
  def element_mult([x|xs],[y|ys],r) do
    [element_mult1(x,y,r)|element_mult(xs,ys,r)]
  end

  def element_mult1([],[],_) do [] end
  def element_mult1([x|xs],[y|ys],r) do
    [x-y*r|element_mult1(xs,ys,r)]
  end

  # for numerical gradient
  def diff([],_,_,_) do [] end
  def diff([m|ms],0,c,d) do
    [diff1(m,0,c,d)|diff(ms,-1,c,d)]
  end
  def diff([m|ms],r,c,d) do
    [m|diff(ms,r-1,c,d)]
  end

  def diff1([],_,_,_) do [] end
  def diff1([v|vs],0,0,d) do
    [v+d|diff1(vs,0,-1,d)]
  end
  def diff1([v|vs],0,c,d) do
    [v|diff1(vs,0,c-1,d)]
  end


  # for CNN
  def convolute(x,y) do
    {r1,c1} = Matrix.size(x)
    {r2,c2} = Matrix.size(y)
    convolute1(x,y,r1-r2+1,c1-c2+1,0,0,1)
  end
  def convolute(x,y,s,p) do
    {r1,c1} = Matrix.size(x)
    {r2,c2} = Matrix.size(y)
    x1 = pad(x,p)
    if rem(r1+2*p-r2,s) == 0 and  rem(c1+2*p-c2,s) == 0 do
      convolute1(x1,y,r1-r2+1,c1-c2+1,0,0,s)
    else
      :error
    end
  end

  def convolute1(_,_,r,_,r,_,_) do [] end
  def convolute1(x,y,r,c,m,n,s) do
    [convolute2(x,y,r,c,m,n,s)|convolute1(x,y,r,c,m+s,n,s)]
  end

  def convolute2(_,_,_,c,_,c,_) do [] end
  def convolute2(x,y,r,c,m,n,s) do
    [convolute_mult_sum(x,y,m,n)|convolute2(x,y,r,c,m,n+s,s)]
  end

  def convolute_mult_sum(x,y,m,n) do
    {r,c} = Matrix.size(y)
    x1 = part(x,m,n,r,c)
    Matrix.emult(x1,y) |> sum
  end

  # padding
  def pad(x,0) do x end
  def pad(x,n) do
    {_,c} = Matrix.size(x)
    zero1 = Matrix.zeros(n,c+n*2)
    zero2 = Matrix.zeros(1,n)
    x1 = Enum.map(x,fn(y) -> hd(zero2) ++ y ++ hd(zero2) end)
    zero1 ++ x1 ++ zero1
  end

  #partial matrix from position(tr,tc) size (m,n)
  def part(x,tr,tc,m,n) do
    {r,c} = Matrix.size(x)
    if tr+m > r or tc+n > c do
      :error
    else
      part1(x,tr,tc,tr+m,n,tr)
    end
  end

  def part1(_,_,_,m,_,m) do [] end
  def part1(x,tr,tc,m,n,r) do
    l = Enum.at(x,r) |> Enum.drop(tc) |> Enum.take(n)
    [l|part1(x,tr,tc,m,n,r+1)]
  end

  # sum of all element
  def sum(x) do
    Enum.reduce(
      Enum.map(x, fn(y) -> Enum.reduce(y, 0, fn(z,acc) -> z + acc end) end),
      0, fn(z,acc) -> z + acc end)
  end

  # for pooling
  def max(x) do
    Enum.max(Enum.map(x, fn(y) -> Enum.max(y) end))
  end
  # poolong
  def pool(x,s) do
    {r,c} = Matrix.size(x)
    if rem(r,s) != 0 or rem(c,s) != 0 do
      :error
    else
      pool1(x,r,c,0,s)
    end
  end

  def pool1(_,r,_,r,_) do [] end
  def pool1(x,r,c,m,s) do
    [pool2(x,r,c,m,0,s)|pool1(x,r,c,m+s,s)]
  end

  def pool2(_,_,c,_,c,_) do [] end
  def pool2(x,r,c,m,n,s) do
    x1 = part(x,m,n,s,s)
    [max(x1)|pool2(x,r,c,m,n+s,s)]
  end

  def rand_matrix(0,_,_) do [] end
  def rand_matrix(m,n,i) do
    [rand_matrix1(n,[],i)|rand_matrix(m-1,n,i)]
  end

  def rand_matrix1(0,res,_) do res end
  def rand_matrix1(n,res,i) do
    rand_matrix1(n-1,[:rand.uniform(i)|res],i)
  end

  # reduce each row vector by sum of each element
  # and calcurate average
  def reduce(x) do
    [reduce1(x)]
  end

  def reduce1([x]) do [x] end
  def reduce1([x,y]) do
    reduce2(x,y)
  end
  def reduce1([x|xs]) do
    reduce2(x,reduce1(xs))
  end

  def reduce2([],[]) do [] end
  def reduce2([x|xs],[y|ys]) do
    [x+y|reduce2(xs,ys)]
  end

  def expand(x,1) do x end
  def expand([x],n) do
    [x|expand([x],n-1)]
  end
end


defmodule MNIST do
  def train_label() do
    {:ok,<<0,0,8,1,0,0,234,96,label::binary>>} = File.read("train-labels-idx1-ubyte")
    label |> String.to_charlist
  end
  def train_image() do
    {:ok,<<0,0,8,3,0,0,234,96,0,0,0,28,0,0,0,28,image::binary>>} = File.read("train-images-idx3-ubyte")
    byte_to_list(image)
  end
  def test_label() do
    {:ok,<<0,0,8,1,0,0,39,16,label::binary>>} = File.read("t10k-labels-idx1-ubyte")
    label |> String.to_charlist
  end
  def test_image() do
    {:ok,<<0,0,8,3,0,0,39,16,0,0,0,28,0,0,0,28,image::binary>>} = File.read("t10k-images-idx3-ubyte")
    byte_to_list(image)
  end

  def byte_to_list(bin) do
    byte_to_list1(bin,784,[],[])
  end

  def byte_to_list1(<<>>,_,ls,res) do
    [Enum.reverse(ls)|res] |> Enum.reverse
  end
  def byte_to_list1(bin,0,ls,res) do
    byte_to_list1(bin,784,[],[Enum.reverse(ls)|res])
  end
  def byte_to_list1(<<b,bs::binary>>,n,ls,res) do
    byte_to_list1(bs,n-1,[b|ls],res)
  end

  def normalize(x,y) do
    Enum.map(x,fn(z) -> z/y end)
  end
  # e.g. 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  def to_onehot(x) do
    to_onehot1(x,9,[])
  end
  def to_onehot1(_,-1,res) do res end
  def to_onehot1(x,x,res) do
    to_onehot1(x,x-1,[1|res])
  end
  def to_onehot1(x,c,res) do
    to_onehot1(x,c-1,[0|res])
  end

  def onehot_to_num([x]) do
    onehot_to_num1(x,0)
  end
  def onehot_to_num1([x|xs],n) do
    if x == Enum.max([x|xs]) do
      n
    else
      onehot_to_num1(xs,n+1)
    end
  end
end


defmodule Worker do
  def part do
    receive do
      {sender,{c,ls1,ls2}} -> send sender,{:answer,[c, Worker.gen_row_vector(ls1,ls2)] }
    end
  end

  def gen_row_vector(_,[]) do [] end
  def gen_row_vector([v],[m|ms]) do
    [inner_product(v,m)|gen_row_vector([v],ms)]
  end

  def inner_product(x,y) do
    inner_product1(x,y,0)
  end

  def inner_product1([],[],res) do res end
  def inner_product1([x|xs],[y|ys],res) do
    inner_product1(xs,ys,x*y+res)
  end
end

defmodule Time do
  defmacro time(exp) do
    quote do
    {time, dict} = :timer.tc(fn() -> unquote(exp) end)
    IO.inspect "time: #{time} micro second"
    IO.inspect "-------------"
    dict
    end
  end

end

defmodule Pmatrix do

  def mult(x,y) do
    y1 = Matrix.transpose(y)
    {r,c} = Matrix.size(x)
    {r1,_} = Matrix.size(y)
    d = 5 # for icore5
    if c != r1 do
      :error
    else if r < 10  do
            Matrix.mult(x,y)
         else
            mult1(x,y1,r,r,lot(r,d),last_lot(r,d))
            mult2(d,[])
            |> Enum.sort
            |> Enum.map(fn(x) -> Enum.drop(x,1) |> hd end)
            |> flatten
        end
    end
  end

  def flatten([]) do [] end
  def flatten([x|xs]) do
    x ++ flatten(xs)
  end

  def lot(m,c) do
    div(m,c)
  end

  def last_lot(m,c) do
    div(m,c) + rem(m,c)
  end

  def mult1(_,_,_,0,_,_) do true end
  def mult1(x,y,m,m,l1,l2) do
    pid = spawn(PWorker,:part,[])
    send pid, {self(),{m,Enum.slice(x,m-l2,l2),y}}
    mult1(x,y,m,m-l2,l1,l2)
  end
  def mult1(x,y,m,c,l1,l2) do
    pid = spawn(PWorker,:part,[])
    send pid, {self(),{c,Enum.slice(x,c-l1,l1),y}}
    mult1(x,y,m,c-l1,l1,l2)
  end


  def mult2(0,res) do res end
  def mult2(d,res) do
    receive do
      {:answer,ls} ->
        mult2(d-1,[ls|res])
    end
  end

  def rand_matrix(0,_,_) do [] end
  def rand_matrix(m,n,i) do
    [rand_matrix1(n,[],i)|rand_matrix(m-1,n,i)]
  end

  def rand_matrix1(0,res,_) do res end
  def rand_matrix1(n,res,i) do
    rand_matrix1(n-1,[:rand.uniform(i)|res],i)
  end

  def rand_matrix_float(0,_) do [] end
  def rand_matrix_float(m,n) do
    [rand_matrix_float1(n,[])|rand_matrix_float(m-1,n)]
  end

  def rand_matrix_float1(0,res) do res end
  def rand_matrix_float1(n,res) do
    rand_matrix_float1(n-1,[:rand.uniform|res])
  end

end

defmodule PWorker do
  def part do
    receive do
      {sender,{c,ls1,ls2}} -> send sender,{:answer,[c, PWorker.gen_row_vector(ls1,ls2)] }
    end
  end

  def gen_row_vector([],_) do [] end
  def gen_row_vector([v|vs],m) do
    [gen_row_vector1(v,m)|gen_row_vector(vs,m)]
  end

  def gen_row_vector1(_,[]) do [] end
  def gen_row_vector1(v,[m|ms]) do
    [inner_product(v,m)|gen_row_vector1(v,ms)]
  end

  def inner_product(x,y) do
    inner_product1(x,y,0)
  end

  def inner_product1([],[],res) do res end
  def inner_product1([x|xs],[y|ys],res) do
    inner_product1(xs,ys,x*y+res)
  end
end
