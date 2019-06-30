defmodule Nat do
  def corpus(str) do
    str |> String.replace("."," .") |> String.split(" ")
  end

  def co_occurence(str) do
    co_occurence1(corpus(str),corpus(str),[])
  end

  def co_occurence1(_,[],_) do [] end
  def co_occurence1(ls,[w|ws],already) do
    if Enum.member?(already,w) do
      [co_occurence2(ls,w,[])|co_occurence1(ls,ws,already)]
    else
      [co_occurence2(ls,w,[])|co_occurence1(ls,ws,[w|already])]
    end
  end

  def co_occurence2([_,"."],_,_) do [0] end
  def co_occurence2(["."],_,_) do [] end
  def co_occurence2([_,w|ls],w,already) do
    [1] ++ co_occurence2([w|ls],w,already)
  end
  def co_occurence2([w,l|ls],w,already) do
    if Enum.member?(already,w) do
      [1] ++ co_occurence2([l|ls],w,already)
    else
      [0,1] ++ co_occurence2(ls,w,[w|already])
    end
  end
  def co_occurence2([_,l2|ls],w,already) do
    [0] ++ co_occurence2([l2|ls],w,already)
  end


end
