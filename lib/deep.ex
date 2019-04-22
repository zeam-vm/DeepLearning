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

  def init_network() do
    [[[0.1,0.3,0.5],[0.2,0.4,0.6]],
     [[0.1,0.2,0.3]],
     fn(x) -> sigmoid(x) end,
     fn(x) -> (1-x)*x end,
     [[0.1,0.4],[0.2,0.5],[0.3,0.6]],
     [[0.1,0.2]],
     fn(x) -> sigmoid(x) end,
     fn(x) -> (1-x)*x end,
     [[0.1,0.3],[0.2,0.4]],
     [[0.1,0.2]],
     fn(x) -> ident(x) end,
     fn(x) -> x end]
  end

  def test_network() do
    [[[0.47355232,0.9977393,0.846680094],
      [0.85557411,0.03560366,0.69422093]],
     [[0.0,0.0,0.0]],
     fn(x) -> softmax(x) end,
     fn(x) -> x end]
  end

  def forward([],x) do x end
  def forward([w,b,f,_|rest],x) do
    x1 = Matrix.mult(x,w)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward(rest,x1)
  end

  def forward_w([],x,_,_,_,_) do x end
  def forward_w([w,b,f,_|rest],x,0,r,c,d) do
    w1 = Dmatrix.diff(w,r,c,d)
    x1 = Matrix.mult(x,w1)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_w(rest,x1,-1,r,c,d)
  end
  def forward_w([w,b,f,_|rest],x,n,r,c,d) do
    x1 = Matrix.mult(x,w)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_w(rest,x1,n-1,r,c,d)
  end

  def forward_b([],x,_,_,_) do x end
  def forward_b([w,b,f,_|rest],x,0,c,d) do
    b1 = Dmatrix.diff(b,0,c,d)
    x1 = Matrix.mult(x,w)|> Matrix.add(b1) |> Enum.map(fn(x) -> f.(x) end)
    forward_b(rest,x1,-1,c,d)
  end
  def forward_b([w,b,f,_|rest],x,n,c,d) do
    x1 = Matrix.mult(x,w)|> Matrix.add(b) |> Enum.map(fn(x) -> f.(x) end)
    forward_b(rest,x1,n-1,c,d)
  end

  def gradient(network,x,t) do
    gradient1(network,x,t,0,network)
  end

  def gradient1([],_,_,_,_) do [] end
  def gradient1([w,b,f,g|rest],x,t,n,network) do
    [gradient_w(w,x,n,network,t),gradient_b(b,x,n,network,t),f,g|
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
    #:io.write(y0)
    #:io.write(y1)
    (cross_entropy(y1,t) - cross_entropy(y0,t)) / h
  end

  def gradient_b(b,x,n,network,t) do
    {_,c} = Matrix.size(b)
    [Enum.map(0..c-1,fn(y1) -> gradient_b1(x,y1,n,network,t) end)]
  end

  def gradient_b1(x,c,n,network,t) do
    h = 0.0001
    y0 = forward(network,x)
    y1 = forward_b(network,x,n,c,h)
    #:io.write(y0)
    #:io.write(y1)
    (cross_entropy(y1,t) - cross_entropy(y0,t)) / h
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
  def learning([w,b,f,g|rest],[w1,b1,_,_|gradrest]) do
    [Dmatrix.element_mult(w,w1),Dmatrix.element_mult(b,b1),f,g|
     learning(rest,gradrest)]
  end

  def sdg() do
    x = [[0.6,0.9]]
    t = [[1,1]]
    sdg1(init_network(),x,t,1)
  end

  def sdg1(network,x,t,n) do
    x1 = forward(network,x)
    error = cross_entropy(x1,t)
    #:io.write(x1)
    if n == 0 do
      network
    else
      network1 = gradient(network,x,t)
      #:io.write(network1)
      #IO.puts("\n")
      network2 = learning(network,network1)
      #:io.write(network2)
      #IO.puts("\n")
      sdg1(network2,x,t,n-1)
    end
  end

end

defmodule Dmatrix do
  def element_mult([],[]) do [] end
  def element_mult([x|xs],[y|ys]) do
    [element_mult1(x,y)|element_mult(xs,ys)]
  end

  def element_mult1([],[]) do [] end
  def element_mult1([x|xs],[y|ys]) do
    [x*y*0.001|element_mult1(xs,ys)]
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
