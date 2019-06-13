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
  # filter
  def parse({:f,_,[x,y]},_) do
    quote do
      {:filter,Dmatrix.new(unquote(x),unquote(y),0.1),1,0.1,Matrix.new(unquote(x),unquote(y))}
    end
  end
  def parse({:f,_,[x,y,lr]},_) do
    quote do
      {:filter,Dmatrix.new(unquote(x),unquote(y),0.1),1,unquote(lr),Matrix.new(unquote(x),unquote(y))}
    end
  end
  def parse({:f,_,[x,y,lr,z]},_) do
    quote do
      {:filter,Dmatrix.new(unquote(x),unquote(y),unquote(z)),1,unquote(lr),Matrix.new(unquote(x),unquote(y))}
    end
  end
  def parse({:f,_,[x,y,lr,z,st]},_) do
    quote do
      {:filter,Dmatrix.new(unquote(x),unquote(y),unquote(z)),unquote(st),unquote(lr),Matrix.new(unquote(x),unquote(y))}
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
      {:weight,unquote(x),0.1,0}
    end
  end
  # constant filter for test
  def parse({:cf,_,[x]},_) do
    quote do
      {:filter,unquote(x),1,0.1,0}
    end
  end
  # constant bias for test
  def parse({:cb,_,[x]},_) do
    quote do
      {:bias,unquote(x),0.1,0}
    end
  end
  # weight
  def parse({:w,_,[x,y]},_) do
    quote do
      {:weight,Dmatrix.new(unquote(x),unquote(y),0.1),0.1,Matrix.new(unquote(x),unquote(y))}
    end
  end
  def parse({:w,_,[x,y,lr]},_) do
    quote do
      {:weight,Dmatrix.new(unquote(x),unquote(y),0.1),unquote(lr),Matrix.new(unquote(x),unquote(y))}
    end
  end
  def parse({:w,_,[x,y,lr,z]},_) do
    quote do
      {:weight,Dmatrix.new(unquote(x),unquote(y),unquote(z)),unquote(lr),Matrix.new(unquote(x),unquote(y))}
    end
  end
  # bias
  def parse({:b,_,[x]},_) do
    quote do
      {:bias,Matrix.new(1,unquote(x)),0.1,Matrix.new(1,unquote(x))}
    end
  end
  def parse({:b,_,[x,lr]},_) do
    quote do
      {:bias,Matrix.new(1,unquote(x)),unquote(lr),Matrix.new(1,unquote(x))}
    end
  end
  # sigmoid
  def parse({:sigmoid,_,nil},_) do
    quote do
      {:function,fn(x) -> DP.sigmoid(x) end,fn(x) -> DP.dsigmoid(x) end}
    end
  end
  # identity
  def parse({:ident,_,nil},_) do
    quote do
      {:function,fn(x) -> DP.ident(x) end,fn(x) -> DP.dident(x) end}
    end
  end
  # relu
  def parse({:relu,_,nil},_) do
    quote do
      {:function,fn(x) -> DP.relu(x) end,fn(x) -> DP.drelu(x) end}
    end
  end
  # softmax
  def parse({:softmax,_,nil},_) do
    quote do
      {:softmax,fn(x) -> DP.softmax(x) end,fn(x) -> DP.dsoftmax(x) end}
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


defmodule BLASNetwork do

  defmacro defnetwork(name, do: body) do
    {_,_,[{arg,_,_}]} = name
    body1 = parse(body,arg)
    quote do
      def unquote(name) do
        unquote(body1)
      end
    end
  end
  # filter
  def parse({:f,_,[x,y]},_) do
    quote do
      {:filter,Cmatrix.new(unquote(x),unquote(y),0.1),1,0.1,Cmatrix.zeros(unquote(x),unquote(y))}
    end
  end
  def parse({:f,_,[x,y,lr]},_) do
    quote do
      {:filter,Cmatrix.new(unquote(x),unquote(y),0.1),1,unquote(lr),Cmatrix.zeros(unquote(x),unquote(y))}
    end
  end
  def parse({:f,_,[x,y,lr,z]},_) do
    quote do
      {:filter,Cmatrix.new(unquote(x),unquote(y),unquote(z)),1,unquote(lr),Cmatrix.zeros(unquote(x),unquote(y))}
    end
  end
  def parse({:f,_,[x,y,lr,z,st]},_) do
    quote do
      {:filter,Cmatrix.new(unquote(x),unquote(y),unquote(z)),unquote(st),unquote(lr),Cmatrix.zeros(unquote(x),unquote(y))}
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
      {:weight,Matrex.new(unquote(x)),0.1,0}
    end
  end
  # constant filter for test
  def parse({:cf,_,[x]},_) do
    quote do
      {:filter,Matrex.new(unquote(x)),1,0.1,0}
    end
  end
  # constant bias for test
  def parse({:cb,_,[x]},_) do
    quote do
      {:bias,Matrex.new(unquote(x)),0.1,0}
    end
  end
  # weight
  def parse({:w,_,[x,y]},_) do
    quote do
      {:weight,Cmatrix.new(unquote(x),unquote(y),0.1),0.1,Cmatrix.zeros(unquote(x),unquote(y))}
    end
  end
  def parse({:w,_,[x,y,lr]},_) do
    quote do
      {:weight,Cmatrix.new(unquote(x),unquote(y),0.1),unquote(lr),Cmatrix.zeros(unquote(x),unquote(y))}
    end
  end
  def parse({:w,_,[x,y,lr,z]},_) do
    quote do
      {:weight,Cmatrix.new(unquote(x),unquote(y),unquote(z)),unquote(lr),Cmatrix.zeros(unquote(x),unquote(y))}
    end
  end
  # bias
  def parse({:b,_,[x]},_) do
    quote do
      {:bias,Cmatrix.zeros(1,unquote(x)),0.1,Cmatrix.zeros(1,unquote(x))}
    end
  end
  def parse({:b,_,[x,lr]},_) do
    quote do
      {:bias,Cmatrix.zeros(1,unquote(x)),unquote(lr),Cmatrix.zeros(1,unquote(x))}
    end
  end
  # sigmoid
  def parse({:sigmoid,_,nil},_) do
    quote do
      {:function,fn(x) -> BLASDP.sigmoid(x) end,fn(x) -> BLASDP.dsigmoid(x) end}
    end
  end
  # identity
  def parse({:ident,_,nil},_) do
    quote do
      {:function,fn(x) -> BLASDP.ident(x) end,fn(x) -> BLASDP.dident(x) end}
    end
  end
  # relu
  def parse({:relu,_,nil},_) do
    quote do
      {:function,fn(x) -> BLASDP.relu(x) end,fn(x) -> BLASDP.drelu(x) end}
    end
  end
  # softmax
  def parse({:softmax,_,nil},_) do
    quote do
      {:softmax,fn(x) -> BLASDP.softmax(x) end,fn(x) -> BLASDP.dsoftmax(x) end}
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
