defmodule DeepTest do
  use ExUnit.Case

  test "forward " do
    assert Deep.test1([[1,2,3]]) == [[22,28]]
    assert Deep.test2([[1,2,3]],0,0,0.1) == [[22.1,28]]
    assert Deep.test2([[1,2,3]],1,1,0.1) == [[22,28.2]]
  end

  test "gradient " do
    assert Deep.test3([[1,2,3]],0,0,[[1,2]]) == 21.005000000002383
  end

  test "test chapter3" do
    #assert Deep.test1() == [[0.7043825919854788, 0.7043825919854788]]
    assert Deep.cross_entropy([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 0.510825457099338
    assert Deep.cross_entropy([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 2.302584092994546
  end
end
