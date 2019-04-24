defmodule DeepTest do
  use ExUnit.Case

  test "test chapter3" do
    assert Deep.test1() == [[0.7043825919854788, 0.7043825919854788]]
    assert Deep.cross_entropy([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 0.510825457099338
    assert Deep.cross_entropy([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 2.302584092994546
  end
end
