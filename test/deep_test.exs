defmodule DLTest do
  use ExUnit.Case

  test "forward " do
    assert Test.test1([[1,2,3]]) == [[22,28]]
    assert Test.test2([[1,2,3]],0,0,0.1) == [[22.1,28]]
    assert Test.test2([[1,2,3]],1,1,0.1) == [[22,28.2]]
  end

  test "gradient " do
    assert Test.test3([[1,2,3]],0,0,[[1,2]]) == 21.005000000002383
  end

  test "test chapter3" do
    #assert DL.test1() == [[0.7043825919854788, 0.7043825919854788]]
    assert DL.cross_entropy([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 0.510825457099338
    assert DL.cross_entropy([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 2.302584092994546
  end
end
