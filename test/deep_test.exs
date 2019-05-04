defmodule DLTest do
  use ExUnit.Case

  test "forward " do
    test_network =
      [[[1,2],
        [3,4],
        [5,6]],
       [[0,0]],
       fn(x) -> DL.ident(x) end,
       fn(x) -> DL.ident(x) end,
       1]
    assert DL.forward(test_network,[[1,2,3]]) == [[22,28]]
    assert DL.forward_w(test_network,[[1,2,3]],0,0,0,0.1) == [[22.1,28]]
    assert DL.forward_w(test_network,[[1,2,3]],0,1,1,0.1) == [[22,28.2]]
    assert DL.apply_function([[1,2,3]], fn(x) -> DL.sigmoid(x) end) == [[0.7310585786300049, 0.8807970779778823, 0.9525741268224334]]

  end


  test "error function" do
    #assert DL.test1() == [[0.7043825919854788, 0.7043825919854788]]
    assert DL.cross_entropy([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 0.510825457099338
    assert DL.cross_entropy([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 2.302584092994546
  end

  test "CNN" do
    a = [[1,2,3,0],
         [0,1,2,3],
         [3,0,1,2],
         [2,3,0,1]]
    b = [[2,0,1],
         [0,1,2],
         [1,0,2]]
    assert Dmatrix.convolute(a,b) == [[15, 16], [6, 15]]
    assert Dmatrix.convolute(a,b,1,0) == [[15, 16], [6, 15]]
    assert Dmatrix.pad([[1,2,3],[2,3,4]],1) == [[0, 0, 0, 0,0], [0, 1, 2, 3, 0], [0, 2, 3, 4, 0], [0, 0, 0, 0,0]]
    assert Dmatrix.pad([[1,2,3],[2,3,4]],0) == [[1, 2, 3], [2, 3, 4]]
    assert Dmatrix.pool([[1,2,1,0],[0,1,2,3],[3,0,1,2],[2,4,0,1]],2) == [[2, 3], [4, 2]]
  end

  test "Dmatrix test" do
    a = Dmatrix.rand_matrix(1,728,3)
    b = Dmatrix.rand_matrix(728,100,3)
    assert Dmatrix.mult(a,b) == Matrix.mult(a,b)
    assert Dmatrix.reduce([[1,2,3],[4,5,6]]) == [5, 7, 9]
    assert Dmatrix.expand([[1,2,3]],3) == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert Dmatrix.expand([[1,2,3]],1) == [[1, 2, 3]]
  end

  test "DLB test" do
    assert DLB.forward(Test.init_network3(),Test.dt3()) == [[0.9524619803383754, 0.9975151146300772],[0.9524619803383751, 0.9975151146300772]]
  end
end
