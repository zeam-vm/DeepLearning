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
    x |> f(2,2) |> sigmoid |> pool(2)
  end

end

# Function Flow
defmodule FF do
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

  # forward for backpropagation
  # this store all middle data
  def forward_for_back(_,[],res) do res end
  def forward_for_back(x,[{:weight,w,_,_}|rest],res) do
    x1 = Pmatrix.mult(w,x)
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

  # numerical gradient
  def numerical_gradient(x,network,t) do
    numerical_gradient1(x,network,t,[],[])
  end

  def numerical_gradient1(_,[],_,_,res) do
    Enum.reverse(res)
  end
  def numerical_gradient1(x,[{:filter,w,st,lr}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,rest)
    numerical_gradient1(x,rest,t,[{:filter,w,st,lr}|before],[w1|res])
  end
  def numerical_gradient1(x,[{:weight,w,lr}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,rest)
    numerical_gradient1(x,rest,t,[{:weight,w,lr}|before],[w1|res])
  end
  def numerical_gradient1(x,[{:bias,w,lr}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,rest)
    numerical_gradient1(x,rest,t,[{:bias,w,lr}|before],[w1|res])
  end
  def numerical_gradient1(x,[y|rest],t,before,res) do
    numerical_gradient1(x,rest,t,[y|before],[y|res])
  end
  # calc numerical gradient of filter,weigth,bias matrix
  def numerical_gradient_matrix(x,w,t,before,rest) do
    1
  end
end
