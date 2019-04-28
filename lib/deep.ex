defmodule Deep do
  @moduledoc """
  Documentation for Deep.
  """

  @doc """
  Hello world.

  ## Examples

      iex> Deep.hello()
      :world

  """
  def foo (x) do
    x+1
  end


  def sigmoid(x) do
    Enum.map(x,fn(y) -> 1 / (1+:math.exp(-y)) end )
  end

  def dsigmoid(x) do
    (1-x)*x
  end

  def step(x) do
    Enum.map(x,fn(y) -> if y > 0 do 1 else 0 end end)
  end

  def relu(x) do
    Enum.map(x,fn(y) -> max(0,y) end)
  end

  def drelu(x) do
    if x > 0 do
      1
    else
      0
    end
  end

  def ident(x) do
    Enum.map(x,fn(y) -> y end)
  end

  def softmax(x) do
    sum = Enum.reduce(x, fn(y, acc) -> :math.exp(y) + acc end)
    Enum.map(x, fn(y) -> :math.exp(y)/sum end)
  end

  def square(x) do
    x*x
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


  def init_network() do
    [[[0.06,0.17,0.12],
      [0.08,0.33,0.18],
      [0.15,0.92,0.12],
      [0.98,0.11,0.20],
      [0.08,0.91,0.12],
      [0.29,0.18,0.21],
      [0.35,0.12,0.22],
      [0.19,0.97,0.03],
      [1.00,0.16,0.93],
      [0.89,0.97,0.11],
      [0.94,0.12,0.09],
      [0.04,0.06,0.13]],
     [[0,0,0]],
     fn(x) -> relu(x) end,
     fn(x) -> drelu(x) end,
     0.01,
     [[0.18,0.92],
      [0.96,0.19],
      [0.92,0.04]],
     [[0,0]],
     fn(x) -> relu(x) end,
     fn(x) -> drelu(x) end,
     0.01]
  end


  def forward([],x) do x end
  def forward([w,b,f,_,_|rest],x) do
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward(rest,x1)
  end

  def bforward(network,x) do
    bforward1(network,x,[x])
  end
  def bforward1([],_,res) do res end
  def bforward1([w,b,f,_,_|rest],x,res) do
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    bforward1(rest,x1,[x1|res])
  end

  def forward_w([],x,_,_,_,_) do x end
  def forward_w([w,b,f,_,_|rest],x,0,r,c,d) do
    w1 = Dmatrix.diff(w,r,c,d)
    x1 = Dmatrix.mult(x,w1)|> Dmatrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_w(rest,x1,-1,r,c,d)
  end
  def forward_w([w,b,f,_,_|rest],x,n,r,c,d) do
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_w(rest,x1,n-1,r,c,d)
  end

  def forward_b([],x,_,_,_) do x end
  def forward_b([w,b,f,_,_|rest],x,0,c,d) do
    b1 = Dmatrix.diff(b,0,c,d)
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b1) |> Enum.map(fn(x) -> f.(x) end)
    forward_b(rest,x1,-1,c,d)
  end
  def forward_b([w,b,f,_,_|rest],x,n,c,d) do
    x1 = Dmatrix.mult(x,w)|> Dmatrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_b(rest,x1,n-1,c,d)
  end

  def gradient(network,x,t) do
    gradient1(network,x,t,0,network)
  end

  def gradient1([],_,_,_,_) do [] end
  def gradient1([w,b,f,g,r|rest],x,t,n,network) do
    [gradient_w(w,x,n,network,t),gradient_b(b,x,n,network,t),f,g,r|
     gradient1(rest,x,t,n+1,network)]
  end

  def gradient_w(w,x,n,network,t) do
    {r,c} = Matrix.size(w)
    Enum.map(0..r-1,
      fn(x1) -> Enum.map(0..c-1,
                  fn(y1) -> gradient_w1(x,x1,y1,n,network,t) end) end)
  end

  def gradient_w1(x,r,c,n,network,t) do
    h = 0.01
    y0 = forward(network,x)
    y1 = forward_w(network,x,n,r,c,h)
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end

  def gradient_b(b,x,n,network,t) do
    {_,c} = Matrix.size(b)
    [Enum.map(0..c-1,fn(y1) -> gradient_b1(x,y1,n,network,t) end)]
  end

  def gradient_b1(x,c,n,network,t) do
    h = 0.01
    y0 = forward(network,x)
    y1 = forward_b(network,x,n,c,h)
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end

  def is_minus(x) do
    Enum.any?(x,fn(x) -> x < 0 end)
  end

  def bgradient(network,x,t) do
    x1 = bforward(network,x)
    l = Matrix.sub(hd(x1),t)
    back(network,l,tl(x1))
  end
  def back(network,l,u) do
    back1(Enum.reverse(network),l,u,[])
  end

  def back1([],_,_,res) do res end
  def back1([r,g,f,_,w|rest],l,[u|us],res) do
    l1 = [Enum.map(hd(l),fn(x) -> g.(x)*x end)]
    l2 = Dmatrix.mult(l1,Matrix.transpose(w))
    w2 = Dmatrix.mult(Matrix.transpose(u),l1)
    back1(rest,l2,us,[w2,l1,f,g,r|res])
  end

  def learning([],_) do [] end
  def learning([w,b,f,g,r|rest],[w1,b1,_,_,_|gradrest]) do
    [Dmatrix.element_mult(w,w1,r),Dmatrix.element_mult(b,b1,r),f,g,r|
     learning(rest,gradrest)]
  end

  def foo() do
    dt1 = [[1,1,1,
           1,0,1,
           1,0,1,
           1,1,1]]
    t1 = [[1,0]]

    network = init_network()
    print_network(bgradient(network,dt1,t1))
    print_network(gradient(network,dt1,t1))
  end

  def sgd() do
    network = init_network()
    dt = Train.dt()
    network1 = mini_batch(network,dt,100)
    :io.write(forward(network1,Enum.at(dt,0)))
    :io.write(Enum.at(dt,1))
    :io.write(forward(network1,Enum.at(dt,14)))
    :io.write(Enum.at(dt,15))
    dt = [[1,1,1,
           0,0,1,
           1,0,1,
           1,1,1]]
    :io.write(forward(network1,dt))
  end

  def mini_batch(network,_,0) do  network end
  def mini_batch(network,dt,n) do
    network1 = mini_batch1(network,dt)
    mini_batch(network1,dt,n-1)
  end

  def mini_batch1(network,[]) do network end
  def mini_batch1(network,[x,t|rest]) do
    #network1 = gradient(network,x,t)
    network1 = bgradient(network,x,t)
    network2 = learning(network,network1)
    mini_batch1(network2,rest)
  end

  def sgd1(network,x,t) do
    x1 = forward(network,x)
    error = mean_square(x1,t)
    IO.puts(error)
    if error < 0.001 do
      network
    else
      network1 = gradient(network,x,t)
      network2 = learning(network,network1)
      sgd1(network2,x,t)
    end
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

  def test_network() do
    [[[1,2],
      [3,4],
      [5,6]],
     [[0,0]],
     fn(x) -> ident(x) end,
     fn(x) -> ident(x) end,
     1]
  end


  def test1(x) do
    network = test_network()
    forward(network,x)
  end
  def test2(x,r,c,d) do
    network = test_network()
    forward_w(network,x,0,r,c,d)
  end
  def test3(x,r,c,t) do
    network = test_network()
    gradient_w1(x,0,r,c,network,t)
  end

end

defmodule Dmatrix do

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
      IO.puts("mult error")
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
      IO.puts("add error")
      :io.write(x)
      :io.write(y)
    else
      Matrix.add(x,y)
    end
  end


  def element_mult([],[],_) do [] end
  def element_mult([x|xs],[y|ys],r) do
    [element_mult1(x,y,r)|element_mult(xs,ys,r)]
  end

  def element_mult1([],[],_) do [] end
  def element_mult1([x|xs],[y|ys],r) do
    [x-x*y*r|element_mult1(xs,ys,r)]
  end

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
end


defmodule Train do
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
