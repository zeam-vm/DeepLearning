# unfinished
defmodule Network do

  defmacro defnetwork(name, do: body) do
    {_,_,[{arg,_,_}]} = name
    body1 = parse(body,arg)
    quote do
      def unquote(name) do
        unquote(arg)
        unquote(body1)
      end
    end
  end
  # filter
  def parse({:f,_,[x,y]},_) do
    quote do
      {:filter,Dmatrix.new(unquote(x),unquote(y),0.1),1,1,0}
    end
  end
  def parse({:f,_,[x,y,z]},_) do
    quote do
      {:filter,Dmatrix.new(unquote(x),unquote(y),unquote(z)),1,1,0}
    end
  end
  def parse({:f,_,[x,y,z,st]},_) do
    quote do
      {:filter,Dmatrix.new(unquote(x),unquote(y),unquote(z)),unquote(st),1,0}
    end
  end
  def parse({:f,_,[x,y,z,st,lr]},_) do
    quote do
      {:filter,Dmatrix.new(unquote(x),unquote(y),unquote(z)),unquote(st),unquote(lr),0}
    end
  end
  # pooling
  def parse({:pool,_,[x]},_) do
    quote do
      {:pooling,unquote(x)}
    end
  end
  # padding
  def parse({:pad,_,[x]},_) do
    quote do
      {:padding,unquote(x)}
    end
  end
  # constant weight for test
  def parse({:cw,_,[x]},_) do
    quote do
      {:weight,unquote(x),1,0}
    end
  end
  # constant filter for test
  def parse({:cf,_,[x]},_) do
    quote do
      {:filter,unquote(x),1,1,0}
    end
  end
  # constant bias for test
  def parse({:cb,_,[x]},_) do
    quote do
      {:bias,unquote(x),1,0}
    end
  end
  # weight
  def parse({:w,_,[x,y]},_) do
    quote do
      {:weight,Dmatrix.new(unquote(x),unquote(y),0.1),1,0}
    end
  end
  def parse({:w,_,[x,y,z]},_) do
    quote do
      {:weight,Dmatrix.new(unquote(x),unquote(y),unquote(z)),1,0}
    end
  end
  def parse({:w,_,[x,y,z,lr]},_) do
    quote do
      {:weight,Dmatrix.new(unquote(x),unquote(y),unquote(z)),unquote(lr),0}
    end
  end
  # bias
  def parse({:b,_,[x]},_) do
    quote do
      {:bias,Matrix.new(1,unquote(x)),1,0}
    end
  end
  def parse({:b,_,[x,lr]},_) do
    quote do
      {:weight,Matrix.new(1,unquote(x)),unquote(lr),0}
    end
  end
  # sigmoid
  def parse({:sigmoid,_,nil},_) do
    quote do
      {:function,fn(x) -> DL.sigmoid(x) end,fn(x) -> DL.dsigmoid(x) end}
    end
  end
  # identity
  def parse({:ident,_,nil},_) do
    quote do
      {:function,fn(x) -> DL.ident(x) end,fn(x) -> DL.dident(x) end}
    end
  end
  # flatten
  def parse({:flatten,_,nil},_) do
    quote do
      {:flatten}
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
  def parse(x,_) do
    :io.write(x)
    IO.puts("Syntax error in defnetwork")
  end

end

defmodule Time do
  defmacro time(exp) do
    quote do
    {time, dict} = :timer.tc(fn() -> unquote(exp) end)
    IO.inspect "time: #{time} micro second"
    IO.inspect "-------------"
    dict
    end
  end

end
