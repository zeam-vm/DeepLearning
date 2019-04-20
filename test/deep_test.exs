defmodule DeepTest do
  use ExUnit.Case

  test "test chapter3" do
    assert Deep.test1() == [[0.3168270764110298, 0.6962790898619668]]
  end
end
