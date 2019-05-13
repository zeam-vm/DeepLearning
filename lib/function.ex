defmodule Foo do
  import Network
  defnetwork init_network1(_x) do
    _x |> w(784,50) |> b(50) |> sigmoid
    |> w(50,100) |> b(100) |> sigmoid
    |> w(100,10) |> b(10) |> sigmoid
  end

  defnetwork init_network2(_x) do
    _x |> f(5,5) |> pool(2) |> flatten
    |> w(144,100) |> b(100) |> sigmoid
    |> w(100,10) |> b(10) |> sigmoid
  end

  defnetwork n1(_x) do
    _x |> w(2,2,0.1)
  end

  defnetwork n2(_x) do
    _x |> cw([[1,2],[2,3]])
  end

  defnetwork n3(_x) do
     _x |> cw([[1,2],[2,3]]) |> sigmoid
  end

  defnetwork n4(_x) do
    _x |> pad(1) |> pool(2) |> cf([[0.1,0.2],[0.3,0.4]])
    |> sigmoid |> flatten
    |> cw([[0.1,0.2,0.3],
           [0.3,0.4,0.5],
           [0.5,0.4,0.2],
           [0.4,0.5,0.3]]) |> b(3) |> sigmoid
  end

  defnetwork n5(_x) do
    _x |> f(3,3) |> flatten
  end

  def dt() do
    [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16]]
  end

  def test() do
    FF.print(FF.numerical_gradient(Foo.dt(),n4(0),[[1,2,3]]))
    FF.newline()
    FF.print(FF.gradient(Foo.dt(),n4(0),[[1,2,3]]))
  end

  def test1() do
    FF.print(FF.forward(dt(),n4(0)))
  end

  def test2() do
    IO.puts("preparing data")
    image = MNIST.train_image()
    network = Foo.init_network2(0)
    image1 = Dmatrix.structure([MNIST.normalize(hd(image),255)],28,28)
    FF.forward(image1,network)
  end

  def test3() do
    FF.gradient(dt(),n5(0),[[1,0],[0,1],[1,1],[0,0]])
  end

end

# Function Flow
defmodule FF do
  def stop() do
    :math.exp(800)
  end

  def print(x) do
    :io.write(x)
  end

  def newline() do
    IO.puts("")
  end

  # apply functin for matrix
  def apply_function([],_) do [] end
  def apply_function([x|xs],f) do
    [Enum.map(x,fn(y) -> f.(y) end)|apply_function(xs,f)]
  end

  # forward
  def forward(x,[]) do x end
  def forward(x,[{:weight,w,_,_}|rest]) do
    x1 = Pmatrix.mult(x,w)
    forward(x1,rest)
  end
  def forward(x,[{:bias,b,_,_}|rest]) do
    x1 = Matrix.add(x,b)
    forward(x1,rest)
  end
  def forward(x,[{:function,f,_}|rest]) do
    x1 = apply_function(x,f)
    forward(x1,rest)
  end
  def forward(x,[{:filter,w,st,_,_}|rest]) do
    x1 = Dmatrix.convolute(x,w,st)
    forward(x1,rest)
  end
  def forward(x,[{:padding,st}|rest]) do
    x1 = Dmatrix.pad(x,st)
    forward(x1,rest)
  end
  def forward(x,[{:pooling,st}|rest]) do
    x1 = Dmatrix.pool(x,st)
    forward(x1,rest)
  end
  def forward(x,[{:flatten}|rest]) do
    x1 = Dmatrix.flatten(x)
    forward(x1,rest)
  end

  # forward for backpropagation
  # this store all middle data
  def forward_for_back(_,[],res) do res end
  def forward_for_back(x,[{:weight,w,_,_}|rest],res) do
    x1 = Pmatrix.mult(x,w)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:bias,b,_,_}|rest],res) do
    x1 = Matrix.add(x,b)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:function,f,_}|rest],res) do
    x1 = apply_function(x,f)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:filter,w,st,_,_}|rest],res) do
    x1 = Dmatrix.convolute(x,w,st)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:padding,st}|rest],res) do
    x1 = Dmatrix.pad(x,st)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:pooling,st}|rest],[_|res]) do
    x1 = Dmatrix.pool(x,st)
    x2 = Dmatrix.sparse(x,st)
    forward_for_back(x1,rest,[x1,x2|res])
  end
  def forward_for_back(x,[{:flatten}|rest],res) do
    x1 = Dmatrix.flatten(x)
    forward_for_back(x1,rest,[x1|res])
  end

  # numerical gradient
  def numerical_gradient(x,network,t) do
    numerical_gradient1(x,network,t,[],[])
  end

  def numerical_gradient1(_,[],_,_,res) do
    Enum.reverse(res)
  end
  def numerical_gradient1(x,[{:filter,w,st,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:filter,w,st,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:filter,w,st,lr,v}|before],[{:filter,w1,st,lr,v}|res])
  end
  def numerical_gradient1(x,[{:weight,w,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:weight,w,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:weight,w1,lr,v}|before],[{:weight,w1,lr,v}|res])
  end
  def numerical_gradient1(x,[{:bias,w,lr}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:bias,w,lr},rest)
    numerical_gradient1(x,rest,t,[{:bias,w,lr}|before],[:bias,w1,lr|res])
  end
  def numerical_gradient1(x,[y|rest],t,before,res) do
    numerical_gradient1(x,rest,t,[y|before],[y|res])
  end
  # calc numerical gradient of filter,weigth,bias matrix
  def numerical_gradient_matrix(x,w,t,before,now,rest) do
    {r,c} = Matrix.size(w)
    Enum.map(0..r-1,
      fn(x1) -> Enum.map(0..c-1,
                  fn(y1) -> numerical_gradient_matrix1(x,t,x1,y1,before,now,rest) end) end)
  end
  def numerical_gradient_matrix1(x,t,r,c,before,{type,w,lr,v},rest) do
    h = 0.0001
    w1 = Dmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (DL.mean_square(y1,t) - DL.mean_square(y0,t)) / h
  end
  def numerical_gradient_matrix1(x,t,r,c,before,{type,w,st,lr,v},rest) do
    h = 0.0001
    w1 = Dmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,st,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,st,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (DL.mean_square(y1,t) - DL.mean_square(y0,t)) / h
  end

  # gradient with backpropagation
  def gradient(x,network,t) do
    x1 = forward_for_back(x,network,[x])
    loss = Matrix.sub(hd(x1),t)
    network1 = Enum.reverse(network)
    backpropagation(loss,network1,tl(x1),[])
  end

  #backpropagation
  def backpropagation(_,[],_,res) do res end
  def backpropagation(l,[{:function,f,g}|rest],[u|ues],res) do
    l1 = Matrix.emult(l,DL.apply_function(u,g))
    backpropagation(l1,rest,ues,[{:function,f,g}|res])
  end
  def backpropagation(l,[{:bias,b,lr,v}|rest],[_|ues],res) do
    backpropagation(l,rest,ues,[{:bias,b,lr,v}|res])
  end
  def backpropagation(l,[{:weight,w,lr,v}|rest],[u|ues],res) do
    w1 = Pmatrix.mult(Matrix.transpose(u),l)
    l1 = Dmatrix.mult(l,Matrix.transpose(w))
    backpropagation(l1,rest,ues,[{:weight,w1,lr,v}|res])
  end
  def backpropagation(l,[{:filter,w,st,lr,v}|rest],[u|ues],res) do
    w1 = Dmatrix.gradient_filter(u,w,l)
    l1 = Dmatrix.deconvolute(u,w,l)
    backpropagation(l1,rest,ues,[{:filter,w1,st,lr,v}|res])
  end
  def backpropagation(l,[{:pooling,st}|rest],[u|ues],res) do
    l1 = Dmatrix.restore(u,l,st)
    backpropagation(l1,rest,ues,[{:pooling,st}|res])
  end
  def backpropagation(l,[{:padding,st}|rest],[_|ues],res) do
    l1 = Dmatrix.remove(l,st)
    backpropagation(l1,rest,ues,[{:padding,st}|res])
  end
  def backpropagation(l,[{:flatten}|rest],[u|ues],res) do
    {r,c} = Matrix.size(u)
    l1 = Dmatrix.structure(l,r,c)
    backpropagation(l1,rest,ues,[{:flatten}|res])
  end

  # update wight and bias
  # learning(network,gradient) -> updated network
  def learning([],_) do [] end
  def learning([{:weight,w,_,_}|rest],[{:weight,w1,lr1,v}|rest1]) do
    [{:weight,Dmatrix.update(w,w1,lr1),lr1,v}|learning(rest,rest1)]
  end
  def learning([{:bias,w,_,_}|rest],[{:bias,w1,lr1,v}|rest1]) do
    [{:bias,Dmatrix.update(w,w1,lr1),lr1,v}|learning(rest,rest1)]
  end
  def learning([{:filter,w,st,_,_}|rest],[{:filter,w1,st,lr1,v}|rest1]) do
    [{:filter,Dmatrix.update(w,w1,lr1),st,lr1,v}|learning(rest,rest1)]
  end
  def learning([network|rest],[_|rest1]) do
    [network|learning(rest,rest1)]
  end

  def sgd(n) do
    IO.puts("preparing data")
    image = MNIST.train_image()
    label = MNIST.train_label()
    network = Foo.init_network2(0)
    image1 = Dmatrix.structure([MNIST.normalize(hd(image),255)],28,28)
    train1 = MNIST.to_onehot(hd(label))
    sgd1([image1],network,[train1],n)
  end

  def sgd1(_,_,_,0) do true end
  def sgd1(image,network,train,n) do
    network1 = gradient(image,network,train)
    network2 = learning(network,network1)
    y = forward(image,network2)
    loss = DL.mean_square(y,train)
    FF.print(loss)
    FF.newline()
    sgd1(image,network2,train,n-1)
  end

end


# function flow for batch_
defmodule FFB do
  # y=result data t=train_data f=error function
  def batch_error(y,t,f) do
    batch_error1(y,t,f,0) / length(y)
  end

  def batch_error1([],[],_,res) do res end
  def batch_error1([y|ys],[t|ts],f,res) do
    batch_error1(ys,ts,f,f.([t],[y])+res)
  end
end
