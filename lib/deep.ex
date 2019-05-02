defmodule Test do
  def init_network1() do
    [Dmatrix.new(784,50),
     Matrix.new(1,50),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     0.5,
     Dmatrix.new(50,100),
     Matrix.new(1,100),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     0.5,
     Dmatrix.new(100,10),
     Matrix.new(1,10),
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     0.5]
  end

  def init_network2() do
    [Dmatrix.new(12,8),
     [[0,0,0,0,0,0,0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     5,
     Dmatrix.new(8,6),
     [[0,0,0,0,0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     5,
     Dmatrix.new(6,3),
     [[0,0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     5,
     Dmatrix.new(3,2),
     [[0,0]],
     fn(x) -> DL.sigmoid(x) end,
     fn(x) -> DL.dsigmoid(x) end,
     5]
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
  @moduledoc """
  Documentation for DL.
  """

  @doc """
  ## Examples



  """
  # activation function
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
  # apply functin for row matrix
  def apply_function([x],f) do
    [Enum.map(x,fn(y) -> f.(y) end)]
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
    l1 = [backpropagation2(hd(l),hd(u1),g)]
    l2 = Dmatrix.mult(l1,Matrix.transpose(w))
    w1 = Dmatrix.mult(Matrix.transpose(u2),l1)
    backpropagation1(rest,l2,us,[w1,l1,f,g,r|res])
  end

  def backpropagation2([],[],_) do [] end
  def backpropagation2([l|ls],[u|us],g) do
    [g.(u)*l|backpropagation2(ls,us,g)]
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

  def mnist(n) do
    image = MNIST.train_image()
    label = MNIST.train_label()
    network = Test.init_network1()
    seq = rand_sequence(100,length(image))
    network1 = batch(network,image,label,100,n,seq)
    test_image = MNIST.test_image()
    test_label = MNIST.test_label()
    seq1 = rand_sequence(10,length(test_image))
    minist1(network1,test_image,test_label,0,seq1)
  end
  # print predict of test data
  def minist1(_,_,_,_,[]) do true end
  def minist1(network,[image|irest],[label|lrest],n,[n|srest]) do
    print(MNIST.onehot_to_num(forward(network,MNIST.normalize(image,255))))
    IO.write(" ")
    print(label)
    newline()
    minist1(network,irest,lrest,n+1,srest)
  end
  def minist1(network,[_|irest],[_|lrest],n,[s|srest]) do
    minist1(network,irest,lrest,n+1,[s|srest])
  end

  def batch(network,_,_,_,0,_) do network end
  def batch(network,image,label,n,c,seq) do
    {network1,error} = batch1(network,image,label,0,seq,0)
    print(c)
    IO.write(" ")
    print(error)
    newline()
    batch(network1,image,label,n,c-1,seq)
  end

  def batch1(network,_,_,_,[],error) do
    {network,error}
  end
  def batch1(network,[image|irest],[label|lrest],n,[n|srest],error) do
    x = MNIST.normalize(image,255)
    t = MNIST.to_onehot(label)
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

defmodule Dmatrix do
  # box-muller rand
  def box_muller() do
    x = :rand.uniform()
    y = :rand.uniform()
    :math.sqrt(-2.0 * :math.log(x)) * :math.cos(2.0 * :math.pi * y);
  end

  #generate initial wweight matrix with box-muller
  def new(0,_) do [] end
  def new(r,c) do
    [new1(c)|new(r-1,c)]
  end

  def new1(0) do [] end
  def new1(c) do
    [box_muller()|new1(c-1)]
  end

  defmacro time(exp) do
    quote do
    {time, dict} = :timer.tc(fn() -> unquote(exp) end)
    IO.inspect "time: #{time} micro second"
    IO.inspect "-------------"
    dict
    end
  end

  # network macro is unfinished
  defmacro network(r,c,f,g,r) do
    m = new(r,c)
    b = Matrix.new(1,c,0)
    quote do
      [unquote(m),
       unquote(b),
       fn(x) -> unquote(f) end,
       fn(x) -> unquote(g) end,
       unquote(r)]
    end
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
    {_,c} = Matrix.size(x)
    {r,_} = Matrix.size(y)
    if r != c do
      IO.puts("Dmatrix mult error")
      :io.write(x)
      :io.write(y)
    else
      Matrix.mult(x,y)
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
    [Enum.map(x,fn(z) -> z/y end)]
  end
  # e.g. 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  def to_onehot(x) do
    [to_onehot1(x,9,[])]
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
