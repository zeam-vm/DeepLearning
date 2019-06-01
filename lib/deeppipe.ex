# Deep Pipe
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

  def softmax([x]) do
    softmax(x)
  end
  def softmax(x) do
    sum = listsum(x)
    Enum.map(x, fn(y) -> :math.exp(y)/sum end)
  end

  defp listsum([]) do 0 end
  defp listsum([l|ls]) do
    :math.exp(l) + listsum(ls)
  end

  # dummy
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

  defp numerical_gradient1(_,[],_,_,res) do
    Enum.reverse(res)
  end
  defp numerical_gradient1(x,[{:filter,w,st,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:filter,w,st,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:filter,w,st,lr,v}|before],[{:filter,w1,st,lr,v}|res])
  end
  defp numerical_gradient1(x,[{:weight,w,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:weight,w,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:weight,w1,lr,v}|before],[{:weight,w1,lr,v}|res])
  end
  defp numerical_gradient1(x,[{:bias,w,lr}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:bias,w,lr},rest)
    numerical_gradient1(x,rest,t,[{:bias,w,lr}|before],[:bias,w1,lr|res])
  end
  defp numerical_gradient1(x,[y|rest],t,before,res) do
    numerical_gradient1(x,rest,t,[y|before],[y|res])
  end
  # calc numerical gradient of filter,weigth,bias matrix
  defp numerical_gradient_matrix(x,w,t,before,now,rest) do
    {r,c} = Matrix.size(w)
    Enum.map(0..r-1,
      fn(x1) -> Enum.map(0..c-1,
                  fn(y1) -> numerical_gradient_matrix1(x,t,x1,y1,before,now,rest) end) end)
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,lr,v},rest) do
    h = 0.0001
    w1 = Dmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,st,lr,v},rest) do
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
  def backpropagation(l,[{:softmax,f,g}|rest],[_|us],res) do
    backpropagation(l,rest,us,[{:softmax,f,g}|res])
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

# Deep Pipe for batch
defmodule DPB do
  # y=result data t=train_data f=error function
  def loss(y,t,:cross) do
    batch_error(y,t,fn(x,y) -> DP.cross_entropy(x,y) end)
  end
  def loss(y,t,:square) do
    batch_error(y,t,fn(x,y) -> DP.mean_square(x,y) end)
  end
  def loss(y,t) do
    batch_error(y,t,fn(x,y) -> DP.mean_square(x,y) end)
  end

  defp batch_error(y,t,f) do
    batch_error1(y,t,f,0) / length(y)
  end

  defp batch_error1([],[],_,res) do res end
  defp batch_error1([y|ys],[t|ts],f,res) do
    batch_error1(ys,ts,f,f.([y],[t])+res)
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
  def forward(x,[{:softmax,f,_}|rest]) do
    x1 = Enum.map(x,f)
    forward(x1,rest)
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
  def forward_for_back(x,[{:softmax,f,_}|rest],res) do
    x1 = Enum.map(x,f)
    forward_for_back(x1,rest,[x1|res])
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
  def numerical_gradient(x,network,t,:square) do
    numerical_gradient(x,network,t)
  end
  def numerical_gradient(x,network,t,:cross) do
    numerical_gradient1(x,network,t,[],[],:cross)
  end
  def numerical_gradient(x,network,t) do
    numerical_gradient1(x,network,t,[],[])
  end

  defp numerical_gradient1(_,[],_,_,res) do
    Enum.reverse(res)
  end
  defp numerical_gradient1(x,[{:filter,w,st,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:filter,w,st,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:filter,w,st,lr,v}|before],[{:filter,w1,st,lr,v}|res])
  end
  defp numerical_gradient1(x,[{:weight,w,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:weight,w,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:weight,w1,lr,v}|before],[{:weight,w1,lr,v}|res])
  end
  defp numerical_gradient1(x,[{:bias,w,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:bias,w,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:bias,w,lr,v}|before],[{:bias,w1,lr,v}|res])
  end
  defp numerical_gradient1(x,[y|rest],t,before,res) do
    numerical_gradient1(x,rest,t,[y|before],[y|res])
  end
  # calc numerical gradient of filter,weigth,bias matrix
  defp numerical_gradient_matrix(x,w,t,before,now,rest) do
    {r,c} = Matrix.size(w)
    Enum.map(0..r-1,
      fn(x1) -> Enum.map(0..c-1,
                  fn(y1) -> numerical_gradient_matrix1(x,t,x1,y1,before,now,rest) end) end)
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,lr,v},rest) do
    h = 0.0001
    w1 = Dmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (batch_error(y1,t,fn(x,y) -> DP.mean_square(x,y) end) - batch_error(y0,t,fn(x,y) -> DP.mean_square(x,y) end)) / h
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,st,lr,v},rest) do
    h = 0.0001
    w1 = Dmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,st,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,st,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (batch_error(y1,t,fn(x,y) -> DP.mean_square(x,y) end) - batch_error(y0,t,fn(x,y) -> DP.mean_square(x,y) end)) / h
  end

  defp numerical_gradient1(_,[],_,_,res,:cross) do
    Enum.reverse(res)
  end
  defp numerical_gradient1(x,[{:filter,w,st,lr,v}|rest],t,before,res,:cross) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:filter,w,st,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:filter,w,st,lr,v}|before],[{:filter,w1,st,lr,v}|res],:cross)
  end
  defp numerical_gradient1(x,[{:weight,w,lr,v}|rest],t,before,res,:cross) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:weight,w,lr,v},rest,:cross)
    numerical_gradient1(x,rest,t,[{:weight,w,lr,v}|before],[{:weight,w1,lr,v}|res],:cross)
  end
  defp numerical_gradient1(x,[{:bias,w,lr,v}|rest],t,before,res,:cross) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:bias,w,lr,v},rest,:cross)
    numerical_gradient1(x,rest,t,[{:bias,w,lr,v}|before],[{:bias,w1,lr,v}|res],:cross)
  end
  defp numerical_gradient1(x,[y|rest],t,before,res,:cross) do
    numerical_gradient1(x,rest,t,[y|before],[y|res],:cross)
  end
  # calc numerical gradient of filter,weigth,bias matrix
  defp numerical_gradient_matrix(x,w,t,before,now,rest,:cross) do
    {r,c} = Matrix.size(w)
    Enum.map(0..r-1,
      fn(x1) -> Enum.map(0..c-1,
                  fn(y1) -> numerical_gradient_matrix1(x,t,x1,y1,before,now,rest,:cross) end) end)
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,lr,v},rest,:cross) do
    h = 0.0001
    w1 = Dmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (batch_error(y1,t,fn(x,y) -> DP.cross_entropy(x,y) end) - batch_error(y0,t,fn(x,y) -> DP.cross_entropy(x,y) end)) / h
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,st,lr,v},rest,:cross) do
    h = 0.0001
    w1 = Dmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,st,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,st,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (batch_error(y1,t,fn(x,y) -> DP.cross_entropy(x,y) end) - batch_error(y0,t,fn(x,y) -> DP.cross_entropy(x,y) end)) / h
  end


  # gradient with backpropagation
  def gradient(x,network,t) do
    x1 = forward_for_back(x,network,[x])
    loss = Matrix.sub(hd(x1),t)
    network1 = Enum.reverse(network)
    backpropagation(loss,network1,tl(x1),[])
  end

  #backpropagation
  defp backpropagation(_,[],_,res) do res end
  defp backpropagation(l,[{:function,f,g}|rest],[u|us],res) do
    if is_tensor(u) do
      l1 = Tensor.emult(l,Tensor.apply_function(u,g))
      backpropagation(l1,rest,us,[{:function,f,g}|res])
    else
      l1 = Matrix.emult(l,DP.apply_function(u,g))
      backpropagation(l1,rest,us,[{:function,f,g}|res])
    end
  end
  defp backpropagation(l,[{:softmax,f,g}|rest],[_|us],res) do
    backpropagation(l,rest,us,[{:softmax,f,g}|res])
  end
  defp backpropagation(l,[{:bias,_,lr,v}|rest],[_|us],res) do
    {n,_} = Matrix.size(l)
    b1 = Dmatrix.reduce(l) |> DP.apply_function(fn(x) -> x/n end)
    backpropagation(l,rest,us,[{:bias,b1,lr,v}|res])
  end
  defp backpropagation(l,[{:weight,w,lr,v}|rest],[u|us],res) do
    {n,_} = Matrix.size(l)
    w1 = Pmatrix.mult(Matrix.transpose(u),l) |> DP.apply_function(fn(x) -> x/n end)
    l1 = Dmatrix.mult(l,Matrix.transpose(w))
    backpropagation(l1,rest,us,[{:weight,w1,lr,v}|res])
  end
  defp backpropagation(l,[{:filter,w,st,lr,v}|rest],[u|us],res) do
    w1 = Tensor.gradient_filter(u,w,l) |> Tensor.average
    l1 = Tensor.deconvolute(u,w,l,st)
    backpropagation(l1,rest,us,[{:filter,w1,st,lr,v}|res])
  end
  defp backpropagation(l,[{:pooling,st}|rest],[u|us],res) do
    l1 = Tensor.restore(u,l,st)
    backpropagation(l1,rest,us,[{:pooling,st}|res])
  end
  defp backpropagation(l,[{:padding,st}|rest],[_|us],res) do
    l1 = Tensor.remove(l,st)
    backpropagation(l1,rest,us,[{:padding,st}|res])
  end
  defp backpropagation(l,[{:flatten}|rest],[u|us],res) do
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


  # print predict of test data
  def accuracy(image,network,label) do
    forward(image,network) |> score(label,0)
  end

  defp score([],[],correct) do correct end
  defp score([x|xs],[l|ls],correct) do
    if MNIST.onehot_to_num(x) == l do
      score(xs,ls,correct+1)
    else
      score(xs,ls,correct)
    end
  end


  # select randome data size of m , range from 0 to n
  def random_select(image,train,m,n) do
    random_select1(image,train,[],[],m,n)
  end

  defp random_select1(_,_,res1,res2,0,_) do {res1,res2} end
  defp random_select1(image,train,res1,res2,m,n) do
    i = :rand.uniform(n)
    image1 = Enum.at(image,i)
    train1 = Enum.at(train,i)
    random_select1(image,train,[image1|res1],[train1|res2],m-1,n)
  end

end
