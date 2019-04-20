defmodule Deep do
  @moduledoc """
  Documentation for Deep.
  """

  @doc """
  Hello world.

  ## Examples

      iex> Deep.hello()
      :world

  """
  #65
  def test1() do
    network = init_network()
    x = [[1.0,0.5]]
    forward(network,x)
  end


  def sigmoid(x) do
    [Enum.map(hd(x),fn(y) -> 1 / (1+:math.exp(-y)) end)]
  end

  def step(x) do
    [Enum.map(hd(x),fn(y) -> if y > 0 do 1 else 0 end end)]
  end

  def relu(x) do
    [Enum.map(hd(x),fn(y) -> min(0,y) end)]
  end

  def init_network() do
    [{:w1,[[0.1,0.3,0.5],[0.2,0.4,0.6]]},
     {:b1,[[0.1,0.2,0.3]]},
     {:w2,[[0.1,0.4],[0.2,0.5],[0.3,0.6]]},
     {:b2,[[0.1,0.2]]},
     {:w3,[[0.1,0.3],[0.2,0.4]]},
     {:b3,[[0.1,0.2]]}]
  end

  def forward(network,x) do
    w1 = network[:w1]
    w2 = network[:w2]
    w3 = network[:w3]
    b1 = network[:b1]
    b2 = network[:b2]
    b3 = network[:b3]
    Matrix.mult(x,w1)|> Matrix.add(b1) |> sigmoid
    |> Matrix.mult(w2) |> Matrix.add(b2) |> sigmoid
    |> Matrix.mult(w3) |> Matrix.add(b3)
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

end
