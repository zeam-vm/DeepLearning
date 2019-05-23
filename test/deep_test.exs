defmodule DLTest do
  use ExUnit.Case


  test "error function" do
    #assert DL.test1() == [[0.7043825919854788, 0.7043825919854788]]
    assert DP.cross_entropy([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 0.510825457099338
    assert DP.cross_entropy([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]) == 2.302584092994546
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
    assert Dmatrix.pad([[1,2,3],[2,3,4]],1) == [[0, 0, 0, 0,0], [0, 1, 2, 3, 0], [0, 2, 3, 4, 0], [0, 0, 0, 0,0]]
    assert Dmatrix.pad([[1,2,3],[2,3,4]],0) == [[1, 2, 3], [2, 3, 4]]
    assert Dmatrix.pool([[1,2,1,0],[0,1,2,3],[3,0,1,2],[2,4,0,1]],2) == [[2, 3], [4, 2]]
    assert Dmatrix.rotate180([[1,2,3],[4,5,6],[7,8,9]]) == [[9,8,7],[6,5,4],[3,2,1]]
    assert Dmatrix.momentum([[1,2,3],[4,5,6]],[[7,8,9],[8,9,10]],0.1) == [[-0.20000000000000007, 0.19999999999999996, 0.6], [1.2, 1.6, 2.0]]
  end

  test "Dmatrix test" do
    a = Dmatrix.rand_matrix(1,728,3)
    b = Dmatrix.rand_matrix(728,100,3)
    assert Dmatrix.mult(a,b) == Matrix.mult(a,b)
    assert Dmatrix.reduce([[1,2,3],[4,5,6]]) == [[5, 7, 9]]
    assert Dmatrix.expand([[1,2,3]],3) == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert Dmatrix.expand([[1,2,3]],1) == [[1, 2, 3]]
  end

  test "FF test" do
    assert DP.forward([[1,2]],Foo.n2(:t)) == [[5, 8]]
  end
end
