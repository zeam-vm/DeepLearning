defmodule Nat do
  def corpus(str) do
    str |> String.replace("."," .") |> String.split(" ")
  end

  def wordvec(x) do
    x
    |> String.to_charlist
    |> Enum.map(fn(x) -> (x-96)/26 end)
  end

end

defmodule Rnn do
  def forward(_,_,[],res) do res end
  def forward([x|xs],h,[wx,wh,b|ls],res) do
    dt = Cmatrix.add(Cmatrix.add(Cmatrix.mult(x,wx),Cmatrix.mult(h,wh)),b)
    dt1 = Cmatrix.apply_function(dt,fn(x) -> :math.tanh(x) end)
    forward(xs,dt1,ls,[dt1|res])
  end
end

#LSTM
defmodule Lstm do
  def forward(_,_,_,[],res) do res end
  def forward([x|xs],h,c,[wx,wh,b|ls],res) do
    dt = Cmatrix.add(Cmatrix.add(Cmatrix.mult(x,wx),Cmatrix.mult(h,wh)),b)
    f = dt |> partf |> Cmatrix.apply_function(fn(x) -> DP.sigmoid(x) end)
    g = dt |> partg |> Cmatrix.apply_function(fn(x) -> :math.tanh(x) end)
    i = dt |> parti |> Cmatrix.apply_function(fn(x) -> DP.sigmoid(x) end)
    o = dt |> parto |> Cmatrix.apply_function(fn(x) -> DP.sigmoid(x) end)
    c1 = Cmatrix.add(Cmatrix.emult(c,f),Cmatrix.emult(g,i))
    h1 = Cmatrix.emult(o,Cmatrix.apply_function(c1,fn(x) -> :math.tanh(x) end))
    forward(xs,h,c1,ls,[h1|res])
  end

  # divide matrex_dt to four parts
  def partf(m) do
    {r,c} = m[:size]
    size = div(c,4)
    index = 1
    Cmatrix.part(m,1,index,r,size)
  end

  def partg(m) do
    {r,c} = m[:size]
    size = div(c,4)
    index = size+1
    Cmatrix.part(m,1,index,r,size)
  end

  def parti(m) do
    {r,c} = m[:size]
    size = div(c,4)
    index = size*2+1
    Cmatrix.part(m,1,index,r,size)
  end

  def parto(m) do
    {r,c} = m[:size]
    size = div(c,4)
    index = size*3+1
    Cmatrix.part(m,1,index,r,size)
  end

end
