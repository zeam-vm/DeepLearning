defmodule Nat do
  def corpus(str) do
    str |> String.replace("."," .") |> String.split(" ")
  end

end
