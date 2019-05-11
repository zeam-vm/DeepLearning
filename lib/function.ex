defmodule Foo do
  import Network
  defnetwork n1(x) do
    x |> w(2,2,0.1)
  end

  defnetwork n2(x) do
    x |> cw([[1,2],[2,3]])
  end

  defnetwork n3(x) do
     x |> cw([[1,2],[2,3]]) |> sigmoid
  end

  defnetwork n4(x) do
    x |> cf([[1.001,2],[3,4]]) |> flatten
  end

  def dt() do
    [[1,2,3],
     [1,2,3],
     [1,2,3]]
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
  def forward_for_back(x,[{:pooling,st}|rest],res) do
    x1 = Dmatrix.pool(x,st)
    x2 = Dmatrix.sparse(x,st)
    forward_for_back(x1,rest,[x2|res])
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
  def learning([{{:weight,w,_,_}}|rest],[{{:weight,w1,lr1,v}}|rest1]) do
    [{:write,Dmatrix.update(w,w1,lr1),lr1,v}|learning(rest,rest1)]
  end
  def learning([{{:bias,w,_,_}}|rest],[{{:bias,w1,lr1,v}}|rest1]) do
    [{:bias,Dmatrix.update(w,w1,lr1),lr1,v}|learning(rest,rest1)]
  end
  def learning([{{:filter,w,st,_,_}}|rest],[{{:filter,w1,st,lr1,v}}|rest1]) do
    [{:filter,Dmatrix.update(w,w1,lr1),st,lr1,v}|learning(rest,rest1)]
  end
  def learning([network|rest],[_,rest1]) do
    [network|learning(rest,rest1)]
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
