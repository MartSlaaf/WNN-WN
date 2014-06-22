__author__ = 'martslaaf'
import numpy as np
from matplotlib.pyplot import plot, show
from wavenet import wavelon_class_constructor, trainer

inp_1 = map(lambda x: float(x), open('/home/martslaaf/Pictures/old_data/nonlinear_xor_1.csv').readlines())
inp_2 = map(lambda x: float(x), open('/home/martslaaf/Pictures/old_data/nonlinear_xor_2.csv').readlines())
inp_3 = map(lambda x: float(x), open('/home/martslaaf/Pictures/old_data/nonlinear_sum.csv').readlines())
outp = map(lambda x: float(x), open('/home/martslaaf/Pictures/old_data/nonlinear_target.csv').readlines())
coun = 1000
tr = []
va = []
for i in xrange(coun-250):
   tr.append((np.array([inp_1[i], inp_2[i], inp_3[i]]),  np.array([outp[i]])))
for i in xrange(coun-250, coun):
   va.append((np.array([inp_1[i], inp_2[i], inp_3[i]]),  np.array([outp[i]])))

n = wavelon_class_constructor(frame=(-200, 200), period=100)
n = n(3, 1, 19)
k = 0
track = trainer(10000, tr, va, n)
outew = []
print track
for j in va:
    outew.append(n.forward(j[0])[0][0])
plot(outp[coun-250: coun], 'g')
plot(outew, 'r')
show()