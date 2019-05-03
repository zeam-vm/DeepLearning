For my study Deep Learning with elixir

Usage:
  iex -S mix
  NL.mnist(batch_size,iteration)

Example:
iex(1)> DL.mnist(1000,200)
prepareing data
c error
200 48.86748226514455
199 50.086982074240616
198 49.89497201235128
197 41.13302243953977
196 31.120977024814618
195 22.869422071261084
194 17.187102781695295
...
8 0.008844112533954122
7 0.008793636809442213
6 0.00874371932026004
5 0.008694350909683702
4 0.008645522619948765
3 0.008597225686872918
2 0.00854945153465225
1 0.008502191770824867
verifying
accuracy rate = 0.7916
:ok
iex(2)>


I implemented backpropagation and numerical-gradient
Now I'm testing small data set.

I implemented CNN partialy.
