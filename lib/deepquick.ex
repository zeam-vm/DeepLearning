# Deep Pipe
defmodule BLASDP do
  def stop() do
    raise("stop")
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

  def listsum([]) do 0 end
  def listsum([x|xs]) do
    :math.exp(x) + listsum(xs)
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

  # apply functin for matrex
  def apply_function(x,f) do
    Matrex.apply(x,f)
  end

  # forward
  def forward(x,[]) do x end
  def forward(x,[{:weight,w,_,_}|rest]) do
    x1 = Cmatrix.mult(x,w)
    forward(x1,rest)
  end
  def forward(x,[{:bias,b,_,_}|rest]) do
    x1 = Cmatrix.add(x,b)
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
    x1 = Cmatrix.convolute(x,w,st)
    forward(x1,rest)
  end
  def forward(x,[{:padding,st}|rest]) do
    x1 = Cmatrix.pad(x,st)
    forward(x1,rest)
  end
  def forward(x,[{:pooling,st}|rest]) do
    x1 = Cmatrix.pool(x,st)
    forward(x1,rest)
  end
  def forward(x,[{:flatten}|rest]) do
    x1 = Cmatrix.flatten(x)
    forward(x1,rest)
  end

  # forward for backpropagation
  # this store all middle data
  def forward_for_back(_,[],res) do res end
  def forward_for_back(x,[{:weight,w,_,_}|rest],res) do
    x1 = Cmatrix.mult(x,w)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:bias,b,_,_}|rest],res) do
    x1 = Cmatrix.add(x,b)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:function,f,_}|rest],res) do
    x1 = apply_function(x,f)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:filter,w,st,_,_}|rest],res) do
    x1 = Cmatrix.convolute(x,w,st)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:padding,st}|rest],res) do
    x1 = Cmatrix.pad(x,st)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:pooling,st}|rest],[_|res]) do
    x1 = Cmatrix.pool(x,st)
    x2 = Cmatrix.sparse(x,st)
    forward_for_back(x1,rest,[x1,x2|res])
  end
  def forward_for_back(x,[{:flatten}|rest],res) do
    x1 = Cmatrix.flatten(x)
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
    {r,c} = w[:size]
    Enum.map(1..r,
      fn(x1) -> Enum.map(1..c,
                  fn(y1) -> numerical_gradient_matrix1(x,t,x1,y1,before,now,rest) end) end)
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,lr,v},rest) do
    h = 0.0001
    w1 = Cmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,st,lr,v},rest) do
    h = 0.0001
    w1 = Cmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,st,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,st,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (mean_square(y1,t) - mean_square(y0,t)) / h
  end

  # gradient with backpropagation
  def gradient(x,network,t) do
    x1 = forward_for_back(x,network,[x])
    loss = Cmatrix.sub(hd(x1),t)
    network1 = Enum.reverse(network)
    backpropagation(loss,network1,tl(x1),[])
  end

  #backpropagation
  def backpropagation(_,[],_,res) do res end
  def backpropagation(l,[{:function,f,g}|rest],[u|us],res) do
    l1 = Cmatrix.emult(l,apply_function(u,g))
    backpropagation(l1,rest,us,[{:function,f,g}|res])
  end
  def backpropagation(l,[{:softmax,f,g}|rest],[_|us],res) do
    backpropagation(l,rest,us,[{:softmax,f,g}|res])
  end
  def backpropagation(l,[{:bias,_,lr,v}|rest],[_|us],res) do
    backpropagation(l,rest,us,[{:bias,l,lr,v}|res])
  end
  def backpropagation(l,[{:weight,w,lr,v}|rest],[u|us],res) do
    w1 = Cmatrix.mult(Cmatrix.transpose(u),l)
    l1 = Cmatrix.mult(l,Cmatrix.transpose(w))
    backpropagation(l1,rest,us,[{:weight,w1,lr,v}|res])
  end
  def backpropagation(l,[{:filter,w,st,lr,v}|rest],[u|us],res) do
    w1 = Cmatrix.gradient_filter(u,w,l)
    l1 = Cmatrix.deconvolute(u,w,l,st)
    backpropagation(l1,rest,us,[{:filter,w1,st,lr,v}|res])
  end
  def backpropagation(l,[{:pooling,st}|rest],[u|us],res) do
    l1 = Cmatrix.restore(u,l,st)
    backpropagation(l1,rest,us,[{:pooling,st}|res])
  end
  def backpropagation(l,[{:padding,st}|rest],[_|us],res) do
    l1 = Cmatrix.remove(l,st)
    backpropagation(l1,rest,us,[{:padding,st}|res])
  end
  def backpropagation(l,[{:flatten}|rest],[u|us],res) do
    {r,c} = u[:size]
    l1 = Cmatrix.structure(l,r,c)
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
