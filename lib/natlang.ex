defmodule Nat do
  def corpus(str) do
    str |> String.replace("."," .") |> String.split(" ")
  end

end

defmodule RNN do
  def forward(_,_,[],res) do res end
  def forward([x|xs],h,[wx,wh,b|ls],res) do
    dt = Cmatrix.add(Cmatrix.add(Cmatrix.mult(x,wx),Cmatrix.mult(h,wh)),b)
    dt1 = Cmatrix.apply_function(dt,fn(x) -> :math.tanh(x) end)
    forward(xs,dt1,ls,[dt1|res])
  end
end
