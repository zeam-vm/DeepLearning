defmodule Foo do
  import Network
  defnetwork init_network1(_x) do
    _x |> w(784,50) |> b(50) |> sigmoid
    |> w(50,100) |> b(100) |> sigmoid
    |> w(100,10) |> b(10) |> sigmoid
  end

  defnetwork init_network2(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300) |> b(300) |> relu
    |> w(300,100) |> b(100) |> relu
    |> w(100,10) |> b(10) |> sigmoid
  end

  defnetwork init_network3(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300) |> b(300) |> relu
    |> w(300,100) |> b(100) |> relu
    |> w(100,10) |> b(10) |> sigmoid
  end

  defnetwork init_network4(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300,0.1,0.001) |> b(300) |> relu
    |> w(300,100,0.1,0.001) |> b(100) |> relu
    |> w(100,10,0.1,0.001) |> b(10) |> sigmoid
  end

  defnetwork init_network5(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300,0.1,0.1) |> b(300) |> relu
    |> w(300,100,0.1,0.1) |> b(100) |> relu
    |> w(100,10,0.1,0.1) |> b(10) |> sigmoid
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
    _x |> f(3,3) |> sigmoid |> flatten
    |> w(4,4) |> b(4) |> sigmoid
  end

  def dt() do
    [
    [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16]],
    [[2,3,1,2],
     [2,3,1,2],
     [3,2,1,1],
     [2,1,1,1]]
    ]
  end

  def tt() do
    [[0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1]]
  end

  def test5() do
    network = n5(0)
    DP.print(DPB.gradient(dt(),network,tt()))
    DP.newline()
    DP.print(DPB.numerical_gradient(dt(),network,tt()))
  end

  def test6() do
    network = n5(0)
    DP.print(DPB.forward(dt(),network))
  end

  def test() do
    DP.print(DP.numerical_gradient(Foo.dt(),n4(0),[[1,2,3]]))
    DP.newline()
    DP.print(DP.gradient(Foo.dt(),n4(0),[[1,2,3]]))
  end

  def test1() do
    DP.print(DP.forward(hd(MNIST.train_image(1)),init_network2(0)))
  end

  def test2() do
    DP.print(DPB.forward(MNIST.train_image(2),init_network2(0)))
  end

  def test3() do
    DPB.gradient(MNIST.train_image(2),init_network2(0),MNIST.train_label_onehot(2))
  end

end

# Function Flow
defmodule DP do
  def stop() do
    :math.exp(800)
  end

  def print(x) do
    :io.write(x)
  end

  def newline() do
    IO.puts("")
  end

  # activate function
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

  # unfinished
  def dsoftmax(x) do
    x
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
  def forward(x,[{:softmax,f,_}|rest]) do
    x1 = f.(x)
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
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end
  def numerical_gradient_matrix1(x,t,r,c,before,{type,w,st,lr,v},rest) do
    h = 0.0001
    w1 = Dmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,st,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,st,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (mean_square(y1,t) - mean_square(y0,t)) / h
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
  def backpropagation(l,[{:function,f,g}|rest],[u|us],res) do
    l1 = Matrix.emult(l,apply_function(u,g))
    backpropagation(l1,rest,us,[{:function,f,g}|res])
  end
  def backpropagation(l,[{:bias,_,lr,v}|rest],[_|us],res) do
    backpropagation(l,rest,us,[{:bias,l,lr,v}|res])
  end
  def backpropagation(l,[{:weight,w,lr,v}|rest],[u|us],res) do
    w1 = Pmatrix.mult(Matrix.transpose(u),l)
    l1 = Dmatrix.mult(l,Matrix.transpose(w))
    backpropagation(l1,rest,us,[{:weight,w1,lr,v}|res])
  end
  def backpropagation(l,[{:filter,w,st,lr,v}|rest],[u|us],res) do
    w1 = Dmatrix.gradient_filter(u,w,l)
    l1 = Dmatrix.deconvolute(u,w,l,st)
    backpropagation(l1,rest,us,[{:filter,w1,st,lr,v}|res])
  end
  def backpropagation(l,[{:pooling,st}|rest],[u|us],res) do
    l1 = Dmatrix.restore(u,l,st)
    backpropagation(l1,rest,us,[{:pooling,st}|res])
  end
  def backpropagation(l,[{:padding,st}|rest],[_|us],res) do
    l1 = Dmatrix.remove(l,st)
    backpropagation(l1,rest,us,[{:padding,st}|res])
  end
  def backpropagation(l,[{:flatten}|rest],[u|us],res) do
    {r,c} = Matrix.size(u)
    l1 = Dmatrix.structure(l,r,c)
    backpropagation(l1,rest,us,[{:flatten}|res])
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

  def online(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(m)
    label = MNIST.train_label(m)
    network = Foo.init_network2(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = online1(image,network,label,m,n)
    correct = accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def online1(_,network,_,_,0) do network end
  def online1(image,network,label,m,n) do
    network1= online2(image,network,label,m)
    train1 = MNIST.to_onehot(hd(label))
    y = forward(hd(image),network1)
    loss = mean_square(y,train1)
    DP.print(loss)
    DP.newline()
    online1(image,network1,label,m,n-1)
  end

  def online2(_,network,_,0) do network end
  def online2([image1|image],network,[label1|label],m) do
    train1 = MNIST.to_onehot(label1)
    network1 = gradient(image1,network,train1)
    network2 = learning(network,network1)
    online2(image,network2,label,m-1)
  end


  # print predict of test data
  def accuracy(_,_,_,0,correct) do
    correct
  end
  def accuracy([image|irest],network,[label|lrest],n,correct) do
    dt = MNIST.onehot_to_num(forward(image,network))
    if dt != label do
      accuracy(irest,network,lrest,n-1,correct)
    else
      accuracy(irest,network,lrest,n-1,correct+1)
    end
  end

  def save(file,network) do
    File.write(file,inspect(network))
  end

  def load(file) do
    Code.eval_file(file) |> elem(0)
  end

end

#----------------------------------------------------------------------------

# function flow for batch_
defmodule DPB do
  # y=result data t=train_data f=error function
  def batch_error(y,t,f) do
    batch_error1(y,t,f,0) / length(y)
  end

  def batch_error1([],[],_,res) do res end
  def batch_error1([y|ys],[t|ts],f,res) do
    batch_error1(ys,ts,f,f.([t],[y])+res)
  end

  # forward
  def is_tensor(x) do
    is_list(x) and is_list(hd(hd(x)))
  end

  def is_matrix(x) do
    is_list(x) and is_number(hd(hd(x)))
  end

  def forward(x,[]) do x end
  def forward(x,[{:weight,w,_,_}|rest]) do
    x1 = Pmatrix.mult(x,w)
    forward(x1,rest)
  end
  def forward(x,[{:bias,b,_,_}|rest]) do
    {r,_} = Matrix.size(x)
    b1 = Dmatrix.expand(b,r)
    x1 = Matrix.add(x,b1)
    forward(x1,rest)
  end
  def forward(x,[{:function,f,_}|rest]) do
    if is_tensor(x) do
      x1 = Tensor.apply_function(x,f)
      forward(x1,rest)
    else
      x1 = DP.apply_function(x,f)
      forward(x1,rest)
    end
  end
  def forward(x,[{:filter,w,st,_,_}|rest]) do
    x1 = Tensor.convolute(x,w,st)
    forward(x1,rest)
  end
  def forward(x,[{:padding,st}|rest]) do
    x1 = Tensor.pad(x,st)
    forward(x1,rest)
  end
  def forward(x,[{:pooling,st}|rest]) do
    x1 = Tensor.pool(x,st)
    forward(x1,rest)
  end
  def forward(x,[{:flatten}|rest]) do
    x1 = Tensor.flatten(x)
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
    {r,_} = Matrix.size(x)
    b1 = Dmatrix.expand(b,r)
    x1 = Matrix.add(x,b1)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:function,f,_}|rest],res) do
    if is_tensor(x) do
      x1 = Tensor.apply_function(x,f)
      forward_for_back(x1,rest,[x1|res])
    else
      x1 = DP.apply_function(x,f)
      forward_for_back(x1,rest,[x1|res])
    end
  end
  def forward_for_back(x,[{:filter,w,st,_,_}|rest],res) do
    x1 = Tensor.convolute(x,w,st)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:padding,st}|rest],res) do
    x1 = Tensor.pad(x,st)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:pooling,st}|rest],[_|res]) do
    x1 = Tensor.pool(x,st)
    x2 = Tensor.sparse(x,st)
    forward_for_back(x1,rest,[x1,x2|res])
  end
  def forward_for_back(x,[{:flatten}|rest],res) do
    x1 = Tensor.flatten(x)
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
    (batch_error(y1,t,fn(x,y) -> DP.mean_square(x,y) end) - batch_error(y0,t,fn(x,y) -> DP.mean_square(x,y) end)) / h
  end
  def numerical_gradient_matrix1(x,t,r,c,before,{type,w,st,lr,v},rest) do
    h = 0.0001
    w1 = Dmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,st,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,st,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (batch_error(y1,t,fn(x,y) -> DP.mean_square(x,y) end) - batch_error(y0,t,fn(x,y) -> DP.mean_square(x,y) end)) / h
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
  def backpropagation(l,[{:function,f,g}|rest],[u|us],res) do
    if is_tensor(u) do
      l1 = Tensor.emult(l,Tensor.apply_function(u,g))
      backpropagation(l1,rest,us,[{:function,f,g}|res])
    else
      l1 = Matrix.emult(l,DP.apply_function(u,g))
      backpropagation(l1,rest,us,[{:function,f,g}|res])
    end
  end
  def backpropagation(l,[{:bias,_,lr,v}|rest],[_|us],res) do
    {n,_} = Matrix.size(l)
    b1 = Dmatrix.reduce(l) |> DP.apply_function(fn(x) -> x/n end)
    backpropagation(l,rest,us,[{:bias,b1,lr,v}|res])
  end
  def backpropagation(l,[{:weight,w,lr,v}|rest],[u|us],res) do
    {n,_} = Matrix.size(l)
    w1 = Pmatrix.mult(Matrix.transpose(u),l) |> DP.apply_function(fn(x) -> x/n end)
    l1 = Dmatrix.mult(l,Matrix.transpose(w))
    backpropagation(l1,rest,us,[{:weight,w1,lr,v}|res])
  end
  def backpropagation(l,[{:filter,w,st,lr,v}|rest],[u|us],res) do
    w1 = Tensor.gradient_filter(u,w,l) |> Tensor.average
    l1 = Tensor.deconvolute(u,w,l,st)
    backpropagation(l1,rest,us,[{:filter,w1,st,lr,v}|res])
  end
  def backpropagation(l,[{:pooling,st}|rest],[u|us],res) do
    l1 = Tensor.restore(u,l,st)
    backpropagation(l1,rest,us,[{:pooling,st}|res])
  end
  def backpropagation(l,[{:padding,st}|rest],[_|us],res) do
    l1 = Tensor.remove(l,st)
    backpropagation(l1,rest,us,[{:padding,st}|res])
  end
  def backpropagation(l,[{:flatten}|rest],[u|us],res) do
    {r,c} = Matrix.size(hd(u))
    l1 = Tensor.structure(l,r,c)
    backpropagation(l1,rest,us,[{:flatten}|res])
  end

  #--------sgd----------
  def learning([],_) do [] end
  def learning([{:weight,w,lr,v}|rest],[{:weight,w1,_,_}|rest1]) do
    [{:weight,Dmatrix.update(w,w1,lr),lr,v}|learning(rest,rest1)]
  end
  def learning([{:bias,w,lr,v}|rest],[{:bias,w1,_,_}|rest1]) do
    [{:bias,Dmatrix.update(w,w1,lr),lr,v}|learning(rest,rest1)]
  end
  def learning([{:filter,w,st,lr,v}|rest],[{:filter,w1,st,_,_}|rest1]) do
    [{:filter,Dmatrix.update(w,w1,lr),st,lr,v}|learning(rest,rest1)]
  end
  def learning([network|rest],[_|rest1]) do
    [network|learning(rest,rest1)]
  end
  #--------momentum-------------
  def learning([],_,:momentum) do [] end
  def learning([{:weight,w,lr,v}|rest],[{:weight,w1,_,_}|rest1],:momentum) do
    v1 = Dmatrix.momentum(v,w1,lr)
    [{:weight,Dmatrix.add(w,v1),lr,v1}|learning(rest,rest1,:momentum)]
  end
  def learning([{:bias,w,lr,v}|rest],[{:bias,w1,_,_}|rest1],:momentum) do
    v1 = Dmatrix.momentum(v,w1,lr)
    [{:bias,Dmatrix.add(w,v1),lr,v1}|learning(rest,rest1,:momentum)]
  end
  def learning([{:filter,w,st,lr,v}|rest],[{:filter,w1,st,_,_}|rest1],:momentum) do
    v1 = Dmatrix.momentum(v,w1,lr)
    [{:filter,Dmatrix.add(w,v1),st,lr,v1}|learning(rest,rest1,:momentum)]
  end
  def learning([network|rest],[_|rest1],:momentum) do
    [network|learning(rest,rest1,:momentum)]
  end
  #--------AdaGrad--------------
  def learning([],_,:adagrad) do [] end
  def learning([{:weight,w,lr,h}|rest],[{:weight,w1,_,_}|rest1],:adagrad) do
    h1 = Matrix.add(h,Matrix.emult(w1,w1))
    [{:weight,Dmatrix.adagrad(w,w1,h1,lr),lr,h1}|learning(rest,rest1,:adagrad)]
  end
  def learning([{:bias,w,lr,h}|rest],[{:bias,w1,_,_}|rest1],:adagrad) do
    h1 = Matrix.add(h,Matrix.emult(w1,w1))
    [{:bias,Dmatrix.adagrad(w,w1,h1,lr),lr,h1}|learning(rest,rest1,:adagrad)]
  end
  def learning([{:filter,w,st,lr,h}|rest],[{:filter,w1,st,_,_}|rest1],:adagrad) do
    h1 = Matrix.add(h,Matrix.emult(w1,w1))
    [{:filter,Dmatrix.adagrad(w,w1,h1,lr),st,lr,h1}|learning(rest,rest1,:adagrad)]
  end
  def learning([network|rest],[_|rest1],:adagrad) do
    [network|learning(rest,rest1,:adagrad)]
  end
  #--------Adam--------------
  def learning([],_,:adam) do [] end
  def learning([{:weight,w,lr,mv}|rest],[{:weight,w1,_,_}|rest1],:adam) do
    mv1 = Dmatrix.adammv(mv,w1)
    [{:weight,Dmatrix.adam(w,mv1,lr),lr,mv1}|learning(rest,rest1,:adam)]
  end
  def learning([{:bias,w,lr,mv}|rest],[{:bias,w1,_,_}|rest1],:adam) do
    mv1 = Dmatrix.adammv(mv,w1)
    [{:bias,Dmatrix.adam(w,mv1,lr),lr,mv1}|learning(rest,rest1,:adam)]
  end
  def learning([{:filter,w,st,lr,mv}|rest],[{:filter,w1,st,_,_}|rest1],:adam) do
    mv1 = Dmatrix.adammv(mv,w1)
    [{:filter,Dmatrix.adam(w,mv1,lr),st,lr,mv1}|learning(rest,rest1,:adam)]
  end
  def learning([network|rest],[_|rest1],:adam) do
    [network|learning(rest,rest1,:adam)]
  end

  # MNIST test
  def batch(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(m)
    label = MNIST.train_label_onehot(m)
    network = Foo.init_network2(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = batch1(image,network,label,n)
    correct = accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def batch1(_,network,_,0) do network end
  def batch1(image,network,train,n) do
    network1 = gradient(image,network,train)
    network2 = learning(network,network1)
    y = forward(image,network1)
    loss = batch_error(y,train,fn(x,y) -> DP.mean_square(x,y) end)
    DP.print(loss)
    DP.newline()
    batch1(image,network2,train,n-1)
  end

  # print predict of test data
  def accuracy(_,_,_,0,correct) do
    correct
  end
  def accuracy([image|irest],network,[label|lrest],n,correct) do
    dt = MNIST.onehot_to_num(DP.forward(image,network))
    if dt != label do
      accuracy(irest,network,lrest,n-1,correct)
    else
      accuracy(irest,network,lrest,n-1,correct+1)
    end
  end

  def sgd(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(2000)
    label = MNIST.train_label_onehot(2000)
    network = Foo.init_network2(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = sgd1(image,network,label,m,n)
    correct = accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def sgd1(_,network,_,_,0) do network end
  def sgd1(image,network,train,m,n) do
    {image1,train1} = random_select(image,train,[],[],m)
    network1 = gradient(image1,network,train1)
    network2 = learning(network,network1)
    y = forward(image1,network2)
    loss = batch_error(y,train1,fn(x,y) -> DP.mean_square(x,y) end)
    DP.print(loss)
    DP.newline()
    sgd1(image,network2,train,m,n-1)
  end

  def momentum(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(2000)
    label = MNIST.train_label_onehot(2000)
    network = Foo.init_network3(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = momentum1(image,network,label,m,n)
    correct = accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def momentum1(_,network,_,_,0) do network end
  def momentum1(image,network,train,m,n) do
    {image1,train1} = random_select(image,train,[],[],m)
    network1 = gradient(image1,network,train1)
    network2 = learning(network,network1,:momentum)
    y = forward(image1,network2)
    loss = batch_error(y,train1,fn(x,y) -> DP.mean_square(x,y) end)
    DP.print(loss)
    DP.newline()
    momentum1(image,network2,train,m,n-1)
  end

  def adagrad(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(2000)
    label = MNIST.train_label_onehot(2000)
    network = Foo.init_network4(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = adagrad1(image,network,label,m,n)
    correct = accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def adagrad1(_,network,_,_,0) do network end
  def adagrad1(image,network,train,m,n) do
    {image1,train1} = random_select(image,train,[],[],m)
    network1 = gradient(image1,network,train1)
    network2 = learning(network,network1,:adagrad)
    y = forward(image1,network2)
    loss = batch_error(y,train1,fn(x,y) -> DP.mean_square(x,y) end)
    DP.print(loss)
    DP.newline()
    adagrad1(image,network2,train,m,n-1)
  end

  def adam(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(2000)
    label = MNIST.train_label_onehot(2000)
    network = Foo.init_network4(0)
    test_image = MNIST.test_image(100)
    test_label = MNIST.test_label(100)
    IO.puts("ready")
    network1 = adam1(image,network,label,m,n)
    correct = accuracy(test_image,network1,test_label,100,0)
    IO.write("accuracy rate = ")
    IO.puts(correct / 100)
  end

  def adam1(_,network,_,_,0) do network end
  def adam1(image,network,train,m,n) do
    {image1,train1} = random_select(image,train,[],[],m)
    network1 = gradient(image1,network,train1)
    network2 = learning(network,network1,:adam)
    y = forward(image1,network2)
    loss = batch_error(y,train1,fn(x,y) -> DP.mean_square(x,y) end)
    DP.print(loss)
    DP.newline()
    adam1(image,network2,train,m,n-1)
  end


  def random_select(_,_,res1,res2,0) do {res1,res2} end
  def random_select(image,train,res1,res2,m) do
    i = :rand.uniform(500)
    image1 = Enum.at(image,i)
    train1 = Enum.at(train,i)
    random_select(image,train,[image1|res1],[train1|res2],m-1)
  end

end
