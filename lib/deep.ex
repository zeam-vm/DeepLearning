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


  #65
  def test1() do
    network = init_network()
    x = [[1.0,0.5]]
    forward(network,x)
  end


  def sigmoid(x) do
    Enum.map(x,fn(y) -> 1 / (1+:math.exp(-y)) end )
  end

  def step(x) do
    Enum.map(x,fn(y) -> if y > 0 do 1 else 0 end end)
  end

  def relu(x) do
    Enum.map(x,fn(y) -> min(0,y) end)
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
    [[[0.1,0.1,0.1],[0.1,0.1,0.1]],
     [[0.1,0.1,0.1]],
     fn(x) -> sigmoid(x) end,
     fn(x) -> (1-x)*x end,
     0.9,
     [[0.1,0.1],[0.1,0.1],[0.1,0.1]],
     [[0.1,0.1]],
     fn(x) -> sigmoid(x) end,
     fn(x) -> (1-x)*x end,
     0.05,
     [[0.1,0.1],[0.1,0.1]],
     [[0.1,0.1]],
     fn(x) -> sigmoid(x) end,
     fn(x) -> (1-x)*x end,
     0.01]
  end


  def forward([],x) do x end
  def forward([w,b,f,_,_|rest],x) do
    x1 = Matrix.mult(x,w)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward(rest,x1)
  end

  def forward_w([],x,_,_,_,_) do x end
  def forward_w([w,b,f,_,_|rest],x,0,r,c,d) do
    w1 = Dmatrix.diff(w,r,c,d)
    x1 = Matrix.mult(x,w1)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_w(rest,x1,-1,r,c,d)
  end
  def forward_w([w,b,f,_,_|rest],x,n,r,c,d) do
    x1 = Matrix.mult(x,w)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_w(rest,x1,n-1,r,c,d)
  end

  def forward_b([],x,_,_,_) do x end
  def forward_b([w,b,f,_,_|rest],x,0,c,d) do
    b1 = Dmatrix.diff(b,0,c,d)
    x1 = Matrix.mult(x,w)|> Matrix.add(b1) |> Enum.map(fn(x) -> f.(x) end)
    forward_b(rest,x1,-1,c,d)
  end
  def forward_b([w,b,f,_,_|rest],x,n,c,d) do
    x1 = Matrix.mult(x,w)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
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
    h = 0.0001
    y0 = forward(network,x)
    y1 = forward_w(network,x,n,r,c,h)
    if is_minus(y0) do
      :io.write(y0)
    end
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end

  def gradient_b(b,x,n,network,t) do
    {_,c} = Matrix.size(b)
    [Enum.map(0..c-1,fn(y1) -> gradient_b1(x,y1,n,network,t) end)]
  end

  def gradient_b1(x,c,n,network,t) do
    h = 0.0001
    y0 = forward(network,x)
    y1 = forward_b(network,x,n,c,h)
    if is_minus(y0) do
      :io.write(y0)
    end
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end

  def is_minus(x) do
    Enum.any?(x,fn(x) -> x < 0 end)
  end

  def back(network,l) do
    back1(Enum.reverse(network),l,[])
  end

  def back1([],_,res) do Enum.reverse(res) end
  def back1([g,f,_,w|rest],l,res) do
    l1 = Enum.map(l,fn(x) -> g.(x) end)
    w1 = Matrix.transpose(w)
    l2 = Matrix.mult(l1,w1)
    w2 = Matrix.mult(Matrix.transpose(l2),l1)
    back1(rest,l2,[g,f,l1,w2|res])
  end

  def learning([],_) do [] end
  def learning([w,b,f,g,r|rest],[w1,b1,_,_,_|gradrest]) do
    [Dmatrix.element_mult(w,w1,r),Dmatrix.element_mult(b,b1,r),f,g,r|
     learning(rest,gradrest)]
  end

  def sgd() do
    x = [[0.5,1]]
    t = [[0.8,0.5]]
    #x1 = [[1,0.5]]
    #t1 = [[0.5,0.8]]
    network = sgd1(init_network(),x,t)
    #network1 = sgd1(network,x1,t1)
    #network2 = sgd1(network1,x,t)
    #network3 = sgd1(network2,x1,t1)
    #network4 = sgd1(network3,x,t)
    #network5 = sgd1(network4,x1,t1)
    :io.write(forward(network,x))
    #:io.write(forward(network5,x1))
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

end

defmodule Dmatrix do
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
