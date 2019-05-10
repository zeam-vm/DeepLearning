# unfinished
defmodule Network do

  defmacro defnetwork(name, do: body) do
    {_,_,[{arg,_,_}]} = name
    body1 = parse(body,arg)
    quote do
      def unquote(name) do
        unquote(body1)
      end
    end
  end

  def parse({:f,_,[x,y,z]},_) do
    quote do
      {:filter,Dmatrix.new(unquote(x),unquote(y),unquote(z))}
    end
  end
  def parse({:sigmoid,_,nil},_) do
    quote do
      {:function,fn(x) -> DL.sigmoid(x) end,fn(x) -> DL.dsigmoid(x) end}
    end
  end
  def parse({:ident,_,nil},_) do
    quote do
      {:function,fn(x) -> DL.ident(x) end,fn(x) -> DL.dident(x) end}
    end
  end
  def parse({x,_,nil},_) do x end
  def parse({:|>,_,exp},arg) do
    parse(exp,arg)
  end
  def parse([{arg,_,nil},exp],arg) do
    [parse(exp,arg)]
  end
  def parse([exp1,exp2],arg) do
    Enum.reverse([parse(exp2,arg)]++Enum.reverse(parse(exp1,arg)))
  end

end

defmodule Foo do
  import Network
  defnetwork n1(_) do
    _ |> f(2,3,0.1) |> sigmoid |> ident |> f(1,1,1)
  end

end
