# paralell gradient
defmodule DPP do
  def gradient(x,network,t) do
    gradient(x,network,t,5)
  end
  def gradient(x,network,t,d) do
    {r,_} = Matrix.size(x)
    if r < 10 do
      DPB.gradient(x,network,t)
    else
      n = div(r,d)
      gradient1(x,network,t,r,n)
      gradient2(r,n,[]) |> sum |> average(r)
    end
  end

  def gradient1(x,network,t,m,n) do
    if m < 2*n do
      pid = spawn(DPPworker,:part,[])
      send pid, {self(),{x,network,t}}
    else
      pid = spawn(DPPworker,:part,[])
      send pid, {self(),{Enum.take(x,n),network,Enum.take(t,n)}}
      gradient1(Enum.drop(x,n),network,Enum.drop(t,n),m-n,n)
    end
  end

  def gradient2(m,n,res) do
      if m < 2*n do
        receive do
          {:answer,ls} -> [ls|res]
        end
      else
        receive do
          {:answer,ls} -> gradient2(m-n,n,[ls|res])
        end
      end
  end

  def average([],_) do [] end
  def average([{:function,f,g}|rest],n) do
    [{:function,f,g}|average(rest,n)]
  end
  def average([{:softmax,f,g}|rest],n) do
    [{:softmax,f,g}|average(rest,n)]
  end
  def average([{:bias,b,lr,v}|rest],n) do
    b1 = DP.apply_function(b,fn(x) -> x/n end)
    [{:bias,b1,lr,v}|average(rest,n)]
  end
  def average([{:weight,w,lr,v}|rest],n) do
    w1 = DP.apply_function(w,fn(x) -> x/n end)
    [{:weight,w1,lr,v}|average(rest,n)]
  end
  def average([{:filter,w,st,lr,v}|rest],n) do
    w1 = DP.apply_function(w,fn(x) -> x/n end)
    [{:filter,w1,st,lr,v}|average(rest,n)]
  end
  def average([{:pooling,st}|rest],n) do
    [{:pooling,st}|average(rest,n)]
  end
  def average([{:padding,st}|rest],n) do
    [{:padding,st}|average(rest,n)]
  end
  def average([{:flatten}|rest],n) do
    [{:flatten}|average(rest,n)]
  end


  def sum([x]) when is_tuple(hd(x)) do x end
  def sum([x|xs]) do
    sum1(x,sum(xs))
  end

  def sum1([],[]) do [] end
  def sum1([{:function,f,g}|rest],[{:function,_,_}|rest1]) do
    [{:function,f,g}|sum1(rest,rest1)]
  end
  def sum1([{:softmax,f,g}|rest],[{:softmax,_,_}|rest1]) do
    [{:softmax,f,g}|sum1(rest,rest1)]
  end
  def sum1([{:bias,b,lr,v}|rest],[{:bias,b1,_,_}|rest1]) do
    [{:bias,Matrix.add(b,b1),lr,v}|sum1(rest,rest1)]
  end
  def sum1([{:weight,w,lr,v}|rest],[{:weight,w1,_,_}|rest1]) do
    [{:weight,Matrix.add(w,w1),lr,v}|sum1(rest,rest1)]
  end
  def sum1([{:filter,w,st,lr,v}|rest],[{:filter,w1,_,_,_}|rest1]) do
    [{:filter,Matrix.add(w,w1),st,lr,v}|sum1(rest,rest1)]
  end
  def sum1([{:pooling,st}|rest],[{:pooling,st}|rest1]) do
    [{:pooling,st}|sum1(rest,rest1)]
  end
  def sum1([{:padding,st}|rest],[{:padding,_}|rest1]) do
    [{:padding,st}|sum1(rest,rest1)]
  end
  def sum1([{:flatten}|rest],[{:flatten}|rest1]) do
    [{:flatten}|sum1(rest,rest1)]
  end

end

# Worker process
defmodule DPPworker do
  def part do
    receive do
      {sender,{x,network,t}} -> send sender,{:answer, DPPworker.gradient(x,network,t)}
    end
  end
  # gradient with backpropagation batch
  def gradient(x,network,t) do
    x1 = DPB.forward_for_back(x,network,[x])
    loss = Matrix.sub(hd(x1),t)
    network1 = Enum.reverse(network)
    backpropagation(loss,network1,tl(x1),[])
  end

  #backpropagation
  def backpropagation(_,[],_,res) do res end
  def backpropagation(l,[{:function,f,g}|rest],[u|us],res) do
    if DPB.is_tensor(u) do
      l1 = Tensor.emult(l,Tensor.apply_function(u,g))
      backpropagation(l1,rest,us,[{:function,f,g}|res])
    else
      l1 = Matrix.emult(l,DP.apply_function(u,g))
      backpropagation(l1,rest,us,[{:function,f,g}|res])
    end
  end
  def backpropagation(l,[{:softmax,f,g}|rest],[_|us],res) do
    backpropagation(l,rest,us,[{:softmax,f,g}|res])
  end
  def backpropagation(l,[{:bias,_,lr,v}|rest],[_|us],res) do
    b1 = Dmatrix.reduce(l)
    backpropagation(l,rest,us,[{:bias,b1,lr,v}|res])
  end
  def backpropagation(l,[{:weight,w,lr,v}|rest],[u|us],res) do
    w1 = Matrix.mult(Matrix.transpose(u),l)
    l1 = Dmatrix.mult(l,Matrix.transpose(w))
    backpropagation(l1,rest,us,[{:weight,w1,lr,v}|res])
  end
  def backpropagation(l,[{:filter,w,st,lr,v}|rest],[u|us],res) do
    w1 = Tensor.gradient_filter(u,w,l) |> Tensor.reduce
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

end
