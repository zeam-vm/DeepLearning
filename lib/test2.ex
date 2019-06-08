defmodule Btest do
  import BLASNetwork
  defnetwork init_network1(_x) do
    _x |> w(2,2) |> b(2) |> softmax
  end

  def test1() do
    network = init_network1(0)
    dt = Matrex.new([[1,2]])
    BLASDP.forward(dt,network)
  end

  def test2() do
    network = init_network1(0)
    dt = Matrex.new([[1,2],[3,4]])
    BLASDPB.forward(dt,network)
  end
end
