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

  test "CNN test" do
    a = [[1,2,3,0],
         [0,1,2,3],
         [3,0,1,2],
         [2,3,0,1]]
    b = [[2,0,1],
         [0,1,2],
         [1,0,2]]
    assert Dmatrix.part(a,2,2,2,2) == [[1,2],[0,1]]
    assert Dmatrix.sparse(a,2) == [[0,2,3,0],[0,0,0,3],[3,0,0,2],[0,3,0,0]]
    assert Dmatrix.convolute(a,b) == [[15, 16], [6, 15]]
    assert Dmatrix.convolute(a,b,1,0) == [[15, 16], [6, 15]]
    assert Dmatrix.pad([[1,2,3],[2,3,4]],1) == [[0, 0, 0, 0,0], [0, 1, 2, 3, 0], [0, 2, 3, 4, 0], [0, 0, 0, 0,0]]
    assert Dmatrix.pad([[1,2,3],[2,3,4]],0) == [[1, 2, 3], [2, 3, 4]]
    assert Dmatrix.pool([[1,2,1,0],[0,1,2,3],[3,0,1,2],[2,4,0,1]],2) == [[2, 3], [4, 2]]
    assert Dmatrix.rotate180([[1,2,3],[4,5,6],[7,8,9]]) == [[9,8,7],[6,5,4],[3,2,1]]
  end

  test "Dmatrix test" do
    a = Dmatrix.rand_matrix(1,728,3)
    b = Dmatrix.rand_matrix(728,100,3)
    assert Dmatrix.mult(a,b) == Matrix.mult(a,b)
    assert Dmatrix.reduce([[1,2,3],[4,5,6]]) == [[5, 7, 9]]
    assert Dmatrix.expand([[1,2,3]],3) == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert Dmatrix.expand([[1,2,3]],1) == [[1, 2, 3]]
  end

  test "DLB test" do

  end

  test "FF test" do
    assert FF.forward([[1,2]],Foo.n2(:t)) == [[5, 8]]
  end
end
