__author__ = 'martslaaf'

import numpy as np
from matplotlib.pyplot import plot, show
from wavenet import wavelon_class_constructor, trainer


coun = 3800
FB = map(lambda x: float(x.split(';')[7]), open('/home/martslaaf/Pictures/Finance/GOOG.csv').readlines()[:3880])
shift_1 = FB[4:coun + 4]
shift_2 = FB[3:coun + 3]
shift_3 = FB[2:coun + 2]
shift_4 = FB[1:coun + 1]
shift_5 = FB[:coun]
no_shift = FB[5:coun + 5]
tr = []
va = []
for i in xrange(coun-1000):
   tr.append((np.array([shift_1[i], shift_2[i], shift_3[i], shift_4[i], shift_5[i]]),  np.array([no_shift[i]])))
for i in xrange(coun-1000, coun):
   va.append((np.array([shift_1[i], shift_2[i], shift_3[i], shift_4[i], shift_5[i]]),  np.array([no_shift[i]])))

n = wavelon_class_constructor(frame=(-200, 200), period=100)
n = n(5, 1, 19)
k = 0
track = trainer(100, tr, va, n)
outew = []
print track
for j in va:
    outew.append(n.forward(j[0])[0][0])
plot(no_shift[coun-1000: coun], 'g')
plot(outew, 'r')
show()