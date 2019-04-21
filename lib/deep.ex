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
    [Enum.map(x,fn(y) -> if y > 0 do 1 else 0 end end)]
  end

  def relu(x) do
    [Enum.map(x,fn(y) -> min(0,y) end)]
  end

  def ident(x) do
     Enum.map(x,fn(y) -> y end)
  end

  #error function
  def cross_entropy([],[]) do 0 end
  def cross_entropy([y|ys],[t|ts]) do
    delta = 1.0e-7
    -(t * :math.log(y+delta)) + cross_entropy(ys,ts)
  end

  def init_network() do
    [[[0.1,0.3,0.5],[0.2,0.4,0.6]],
     [[0.1,0.2,0.3]],
     fn(x) -> sigmoid(x) end,
     [[0.1,0.4],[0.2,0.5],[0.3,0.6]],
     [[0.1,0.2]],
     fn(x) -> sigmoid(x) end,
     [[0.1,0.3],[0.2,0.4]],
     [[0.1,0.2]],
     fn(x) -> ident(x) end]
  end

  def forward([],x) do x end
  def forward([w,b,f|rest],x) do
    x1 = Matrix.mult(x,w)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward(rest,x1)
  end

  def forward_w([],x,_,_,_,_) do x end
  def forward_w([w,b,f|rest],x,0,r,c,d) do
    w1 = Dmatrix.diff(w,r,c,d)
    x1 = Matrix.mult(x,w1)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_w(rest,x1,-1,r,c,d)
  end
  def forward_w([w,b,f|rest],x,n,r,c,d) do
    x1 = Matrix.mult(x,w)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_w(rest,x1,n-1,r,c,d)
  end

  def forward_b([],x,_,_,_) do x end
  def forward_b([w,b,f|rest],x,0,c,d) do
    b1 = Dmatrix.diff(b,0,c,d)
    x1 = Matrix.mult(x,w)|> Matrix.add(b1) |> Enum.map(fn(x) -> f.(x) end)
    forward_b(rest,x1,-1,c,d)
  end
  def forward_b([w,b,f|rest],x,n,c,d) do
    x1 = Matrix.mult(x,w)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_b(rest,x1,n-1,c,d)
  end

  def gradient(network,x,t) do
    gradient1(network,x,t,0,network)
  end

  def gradient1([w,b,f|rest],x,t,n,network) do
    [gradient_w(w,x,n,network,t),gradient_b(b,x,n,network,t),f|
     gradient1(rest,x,t,n-1,network)]
  end

  def gradient_w(w,x,n,network,t) do
    {r,c} = Matrix.size(w)
    Enum.map(0..r-1,
      fn(x1) -> Enum.map(0..c-1,
                  fn(y1) -> gradient_w1(x,x1,y1,n,network,t) end) end)
  end

  def gradient_w1(x,r,c,n,network,t) do
    y = forward_w(network,x,n,r,c,0.0001)
    cross_entropy(y,t)
  end

  def gradient_b(b,x,n,network,t) do
    {_,c} = Matrix.size(b)
    Enum.map(0..c-1,fn(y1) -> gradient_b1(x,y1,n,network,t) end)
  end

  def gradient_b1(x,c,n,network,t) do
    y = forward_b(network,x,n,c,0.0001)
    cross_entropy(y,t)
  end

end

defmodule Dmatrix do
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


  def mult(x,y,r,c,d) do
    mult1(x,Matrix.transpose(y),c,r,d)
  end

  def mult1([],_,_,_,_) do [] end
  def mult1([x|xs],[y|ys],r,c,d) do
    [mult2(x,[y|ys],r,c,d)|mult1(xs,[y|ys],r,c,d)]
  end

  def mult2(_,[],_,_,_) do [] end
  def mult2(x,[y|ys],0,c,d) do
    [inner_product_diff(x,y,c,d)|mult2(x,ys,-1,c,d)]
  end
  def mult2(x,[y|ys],r,c,d) do
    [inner_product(x,y)|mult2(x,ys,r-1,c,d)]
  end


  def inner_product([],[]) do 0 end
  def inner_product([x|xs],[y|ys]) do
    x*y + inner_product(xs,ys)
  end

  def inner_product_diff([],[],_,_) do 0 end
  def inner_product_diff([x|xs],[y|ys],0,d) do
    x*(y+d) + inner_product_diff(xs,ys,-1,d)
  end
  def inner_product_diff([x|xs],[y|ys],n,d) do
    x*y + inner_product_diff(xs,ys,n-1,d)
  end

  def add([],[],_,_,_) do [] end
  def add([x|xs],[y|ys],r,c,d) do
    [add1(x,y,r,c,d)|add(xs,ys,r-1,c,d)]
  end

  def add1([],[],_,_,_) do [] end
  def add1([x|xs],[y|ys],0,0,d) do
    [x+(y+d)|add1(xs,ys,0,-1,d)]
  end
  def add1([x|xs],[y|ys],0,c,d) do
    [x+y|add1(xs,ys,0,c-1,d)]
  end
  def add1([x|xs],[y|ys],r,c,d) do
    [x+y|add1(xs,ys,r,c,d)]
  end



  def is_matrix(x) do
    if is_list(x) and is_list(hd(x)) and length(x) >= 2 do
      true
    else
      false
    end
  end

  def is_vector(x) do
    if is_list(x) and is_list(hd(x)) and length(x) == 1 do
      true
    else
      false
    end
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
