defmodule Tensor do

  def reduce([x]) do x end
  def reduce([x|xs]) do
    Matrix.add(x,reduce(xs))
  end

  def average_for_pool(x) do
    y = Dmatrix.notzero_to_one(x)
    reduce(x) |> Dmatrix.ediv(y)
  end

  # normal average
  def average(x) do
    n = length(x)
    reduce(x) |> FF.apply_function(fn(y) -> y/n end)
  end

  # op(tensor,matrix)
  def apply_operation([],_,_) do [] end
  def apply_operation([x|xs],y,op) do
    [op.(x,y)|apply_operation(xs,y,op)]
  end

  # apply_function(tensor,function)
  def apply_function([],_) do [] end
  def apply_function([x|xs],f) do
    [FF.apply_function(x,f)|apply_function(xs,f)]
  end

end



defmodule Dmatrix do
  # divide each element of x by y
  def ediv([],[]) do [] end
  def ediv([x|xs],[y|ys]) do
    [ediv1(x,y)|ediv(xs,ys)]
  end

  def ediv1([],[]) do [] end
  def ediv1([x|xs],[y|ys]) do
    [x/y|ediv1(xs,ys)]
  end

  # change notzero element to 1
  # element 0 is 0
  def notzero_to_one([]) do [] end
  def notzero_to_one([x|xs]) do
    x1 = Enum.map(x,fn(y) -> if y != 0 do 1 else 0 end end)
    [x1|notzero_to_one(xs)]
  end
  # box-muller rand
  def box_muller() do
    x = :rand.uniform()
    y = :rand.uniform()
    :math.sqrt(-2.0 * :math.log(x)) * :math.cos(2.0 * :math.pi * y);
  end

  #generate initial weight matrix with box-muller
  def new(0,_,_) do [] end
  def new(r,c,rate) do
    [new1(c,rate)|new(r-1,c,rate)]
  end

  def new1(0,_) do [] end
  def new1(c,rate) do
    [box_muller()*rate|new1(c-1,rate)]
  end

  def print([]) do
    IO.puts("")
  end
  def print([x|xs]) do
    :io.write(x)
    IO.puts("")
    print(xs)
  end

  def mult(x,y) do
    {_,c} = Matrix.size(x)
    {r,_} = Matrix.size(y)
    if r != c do
      IO.puts("Dmatrix mult error")
      :io.write(x)
      :io.write(y)
    else
      Matrix.mult(x,y)
    end
  end

  def add(x,y) do
    {r1,c1} = Matrix.size(x)
    {r2,c2} = Matrix.size(y)
    if r1 != r2 or c1 != c2 do
      IO.puts("Dmatrix add error")
      :io.write(x)
      :io.write(y)
    else
      Matrix.add(x,y)
    end
  end

  # for learning
  # each element x-y*r
  def update([],[],_) do [] end
  def update([x|xs],[y|ys],r) do
    [update1(x,y,r)|update(xs,ys,r)]
  end

  def update1([],[],_) do [] end
  def update1([x|xs],[y|ys],r) do
    [x-y*r|update1(xs,ys,r)]
  end

  # for numerical gradient
  # add d to element (r,c)
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


  # for CNN
  # convolution
  def convolute(x,y) do
    {r1,c1} = Matrix.size(x)
    {r2,c2} = Matrix.size(y)
    convolute1(x,y,r1-r2+1,c1-c2+1,0,0,1)
  end
  def convolute(x,y,s) do
    {r1,c1} = Matrix.size(x)
    {r2,c2} = Matrix.size(y)
    if rem(r1-r2,s) == 0 and  rem(c1-c2,s) == 0 do
      convolute1(x,y,r1-r2+1,c1-c2+1,0,0,s)
    else
      :error
    end
  end
  def convolute(x,y,s,p) do
    {r1,c1} = Matrix.size(x)
    {r2,c2} = Matrix.size(y)
    x1 = pad(x,p)
    if rem(r1+2*p-r2,s) == 0 and  rem(c1+2*p-c2,s) == 0 do
      convolute1(x1,y,r1-r2+1,c1-c2+1,0,0,s)
    else
      :error
    end
  end

  def convolute1(_,_,r,_,r,_,_) do [] end
  def convolute1(x,y,r,c,m,n,s) do
    [convolute2(x,y,r,c,m,n,s)|convolute1(x,y,r,c,m+s,n,s)]
  end

  def convolute2(_,_,_,c,_,c,_) do [] end
  def convolute2(x,y,r,c,m,n,s) do
    [convolute_mult_sum(x,y,m,n)|convolute2(x,y,r,c,m,n+s,s)]
  end

  def convolute_mult_sum(x,y,m,n) do
    {r,c} = Matrix.size(y)
    x1 = part(x,m,n,r,c)
    Matrix.emult(x1,y) |> sum
  end

  # padding
  def pad(x,0) do x end
  def pad(x,n) do
    {_,c} = Matrix.size(x)
    zero1 = Matrix.zeros(n,c+n*2)
    zero2 = Matrix.zeros(1,n)
    x1 = Enum.map(x,fn(y) -> hd(zero2) ++ y ++ hd(zero2) end)
    zero1 ++ x1 ++ zero1
  end

  #remove ,-> padding
  def remove(x,0) do x end
  def remove(x,n) do
    x1 = Enum.drop(Enum.reverse(Enum.drop(Enum.reverse(x),n)),n)
    Enum.map(x1,fn(y) -> Enum.drop(Enum.reverse(Enum.drop(Enum.reverse(y),n)),n) end)
  end

  #partial matrix from position(tr,tc) size (m,n)
  def part(x,tr,tc,m,n) do
    {r,c} = Matrix.size(x)
    if tr+m > r or tc+n > c do
      :error
    else
      part1(x,tr,tc,tr+m,n,tr)
    end
  end

  def part1(_,_,_,m,_,m) do [] end
  def part1(x,tr,tc,m,n,r) do
    l = Enum.at(x,r) |> Enum.drop(tc) |> Enum.take(n)
    [l|part1(x,tr,tc,m,n,r+1)]
  end

  # sum of all element
  def sum(x) do
    Enum.reduce(
      Enum.map(x, fn(y) -> Enum.reduce(y, 0, fn(z,acc) -> z + acc end) end),
      0, fn(z,acc) -> z + acc end)
  end

  # for pooling
  def max(x) do
    Enum.max(Enum.map(x, fn(y) -> Enum.max(y) end))
  end
  # poolong
  def pool(x,s) do
    {r,c} = Matrix.size(x)
    if rem(r,s) != 0 or rem(c,s) != 0 do
      IO.puts("Bad argment pooling")
      :error
    else
      pool1(x,r,c,0,s)
    end
  end

  def pool1(_,r,_,r,_) do [] end
  def pool1(x,r,c,m,s) do
    [pool2(x,r,c,m,0,s)|pool1(x,r,c,m+s,s)]
  end

  def pool2(_,_,c,_,c,_) do [] end
  def pool2(x,r,c,m,n,s) do
    x1 = part(x,m,n,s,s)
    [max(x1)|pool2(x,r,c,m,n+s,s)]
  end

  # restore <-> pool
  def restore(u,los,st) do
    Matrix.emult(u,increase(los,st))
  end

  # e.g.  increase(x,2)   [[1,1,2,2,],
  #        [[1,2],         [1,1,2,2],
  #         [3,4]]     ->  [3,3,4,4],
  #                        [3,3,4,4]]
  def increase([],_) do [] end
  def increase([x|xs],st) do
    x1 = increase1(x,st) |> increase2(st)
    x1 ++ increase(xs,st)
  end

  def increase1([],_) do [] end
  def increase1([x|xs],st) do
    increase3(x,st) ++ increase1(xs,st)
  end

  def increase2(_,0) do [] end
  def increase2(x,st) do
    [x|increase2(x,st-1)]
  end

  def increase3(x,1) do [x] end
  def increase3(x,st) do
    [x|increase3(x,st-1)]
  end

  # sparse for matrix (use backpropagation)
  def sparse(x,s) do
    {r,c} = Matrix.size(x)
    if rem(r,s) != 0 or rem(c,s) != 0 do
      :error
    else
      sparse1(x,r,c,0,s)
    end
  end

  def sparse1(_,r,_,r,_) do [] end
  def sparse1(x,r,c,m,s) do
    [sparse2(x,r,c,m,0,s)|sparse1(x,r,c,m+s,s)]
  end

  def sparse2(_,_,c,_,c,_) do [] end
  def sparse2(x,r,c,m,n,s) do
    x1 = part(x,m,n,s,s)
    m = max(x1)
    x2 = FF.apply_function(x1,fn(y) -> if y==m do m else 0 end end)
    [x2|sparse2(x,r,c,m,n+s,s)]
  end


  def rotate180(x) do
    Enum.reverse(Enum.map(x,fn(y) -> Enum.reverse(y) end))
  end

  def deconvolute(u,filter,loss) do
    loss |> pad(1) |> convolute(rotate180(filter)) |> Matrix.emult(u)
  end

  def gradient_filter(u,filter,loss) do
    {r,c} = Matrix.size(filter)
    {m,n} = Matrix.size(loss)
    Enum.map(0..r-1,
      fn(x1) -> Enum.map(0..c-1,
                  fn(y1) -> gradient_filter1(u,filter,loss,x1,y1,m,n) end) end)
  end

  def gradient_filter1(u,filter,error,x1,y1,m,n) do
    p = part(u,x1,y1,m,n)
    p |> Matrix.emult(error)
    |> DL.apply_function(fn(y) -> y * Matrix.elem(filter,x1,y1) end)
    |> Dmatrix.sum
  end


  def rand_matrix(0,_,_) do [] end
  def rand_matrix(m,n,i) do
    [rand_matrix1(n,[],i)|rand_matrix(m-1,n,i)]
  end

  def rand_matrix1(0,res,_) do res end
  def rand_matrix1(n,res,i) do
    rand_matrix1(n-1,[:rand.uniform(i)|res],i)
  end

  # reduce each row vector by sum of each element
  def reduce(x) do
    [reduce1(x)]
  end

  def reduce1([x]) do [x] end
  def reduce1([x,y]) do
    reduce2(x,y)
  end
  def reduce1([x|xs]) do
    reduce2(x,reduce1(xs))
  end

  def reduce2([],[]) do [] end
  def reduce2([x|xs],[y|ys]) do
    [x+y|reduce2(xs,ys)]
  end

  def expand(x,1) do x end
  def expand([x],n) do
    [x|expand([x],n-1)]
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

  def normalize(x,y) do
    Enum.map(x,fn(z) -> z/y end)
  end
  # e.g. 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  def to_onehot(x) do
    to_onehot1(x,9,[])
  end
  def to_onehot1(_,-1,res) do res end
  def to_onehot1(x,x,res) do
    to_onehot1(x,x-1,[1|res])
  end
  def to_onehot1(x,c,res) do
    to_onehot1(x,c-1,[0|res])
  end

  def onehot_to_num([x]) do
    onehot_to_num1(x,0)
  end
  def onehot_to_num1([x|xs],n) do
    if x == Enum.max([x|xs]) do
      n
    else
      onehot_to_num1(xs,n+1)
    end
  end
end


defmodule Worker do
  def part do
    receive do
      {sender,{c,ls1,ls2}} -> send sender,{:answer,[c, Worker.gen_row_vector(ls1,ls2)] }
    end
  end

  def gen_row_vector(_,[]) do [] end
  def gen_row_vector([v],[m|ms]) do
    [inner_product(v,m)|gen_row_vector([v],ms)]
  end

  def inner_product(x,y) do
    inner_product1(x,y,0)
  end

  def inner_product1([],[],res) do res end
  def inner_product1([x|xs],[y|ys],res) do
    inner_product1(xs,ys,x*y+res)
  end
end


defmodule Pmatrix do

  def mult(x,y) do
    y1 = Matrix.transpose(y)
    {r,c} = Matrix.size(x)
    {r1,_} = Matrix.size(y)
    d = 5 # for icore5
    if c != r1 do
      :io.write(x)
      :io.write(y)
      :error
    else if r < 10  do
            Matrix.mult(x,y)
         else
            mult1(x,y1,r,r,lot(r,d),last_lot(r,d))
            mult2(d,[])
            |> Enum.sort
            |> Enum.map(fn(x) -> Enum.drop(x,1) |> hd end)
            |> flatten
        end
    end
  end

  def flatten([]) do [] end
  def flatten([x|xs]) do
    x ++ flatten(xs)
  end

  def lot(m,c) do
    div(m,c)
  end

  def last_lot(m,c) do
    div(m,c) + rem(m,c)
  end

  def mult1(_,_,_,0,_,_) do true end
  def mult1(x,y,m,m,l1,l2) do
    pid = spawn(PWorker,:part,[])
    send pid, {self(),{m,Enum.slice(x,m-l2,l2),y}}
    mult1(x,y,m,m-l2,l1,l2)
  end
  def mult1(x,y,m,c,l1,l2) do
    pid = spawn(PWorker,:part,[])
    send pid, {self(),{c,Enum.slice(x,c-l1,l1),y}}
    mult1(x,y,m,c-l1,l1,l2)
  end


  def mult2(0,res) do res end
  def mult2(d,res) do
    receive do
      {:answer,ls} ->
        mult2(d-1,[ls|res])
    end
  end

  def rand_matrix(0,_,_) do [] end
  def rand_matrix(m,n,i) do
    [rand_matrix1(n,[],i)|rand_matrix(m-1,n,i)]
  end

  def rand_matrix1(0,res,_) do res end
  def rand_matrix1(n,res,i) do
    rand_matrix1(n-1,[:rand.uniform(i)|res],i)
  end

  def rand_matrix_float(0,_) do [] end
  def rand_matrix_float(m,n) do
    [rand_matrix_float1(n,[])|rand_matrix_float(m-1,n)]
  end

  def rand_matrix_float1(0,res) do res end
  def rand_matrix_float1(n,res) do
    rand_matrix_float1(n-1,[:rand.uniform|res])
  end

end

defmodule PWorker do
  def part do
    receive do
      {sender,{c,ls1,ls2}} -> send sender,{:answer,[c, PWorker.gen_row_vector(ls1,ls2)] }
    end
  end

  def gen_row_vector([],_) do [] end
  def gen_row_vector([v|vs],m) do
    [gen_row_vector1(v,m)|gen_row_vector(vs,m)]
  end

  def gen_row_vector1(_,[]) do [] end
  def gen_row_vector1(v,[m|ms]) do
    [inner_product(v,m)|gen_row_vector1(v,ms)]
  end

  def inner_product(x,y) do
    inner_product1(x,y,0)
  end

  def inner_product1([],[],res) do res end
  def inner_product1([x|xs],[y|ys],res) do
    inner_product1(xs,ys,x*y+res)
  end
end
