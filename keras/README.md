Results
2dim (input {#solids,#stripes}, output winner)
made output a 2d softmax instead of 1d sigmoid to see if any difference. there wasn't.
val loss and val acc: .4142, .6013

6dim (input {#easy solids,#easy stripes,#med solids,#med stripes,#hard solids,#hard stripes})
with d2 zoning: .4517, .6111
with new diff, changed the zones, fixed a bug
loss and acc: .3989, .6296

Pip (input {xy-coords for each solid, then each stripe, then cue and eight})
(data is diff fn for each ball)
loss and acc: .3730, .6288


Duncan (input xy-coords for each ball, output local advantage {-1,0,1})
without conv1d layer: .4026, .3615
with conv1d layer: .3646, .4670

Duncanp (same as duncan but with d1/d2/theta instead of x/y)
with conv1d: .3956, .4231


Simple (input xy coords of a ball, output difficulty)
loss and acc: .1151, .0372
After adding 3 layers, giving more data, shuffling data, and increasing bs: .0256. It’s just underfitting. Difficulty function is still not well-smoothed.
