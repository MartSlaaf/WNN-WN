__author__ = 'martslaaf'
import numpy as np
from matplotlib.pyplot import plot, show
from wavenet import wavelon_class_constructor, trainer


coun = 3000
inp_s = map(lambda x: float(x.split(',')[0]), open('/home/martslaaf/Learn_Coding/pybrain/sig.csv').readlines())
outp_s = map(lambda x: float(x.split(',')[1]), open('/home/martslaaf/Learn_Coding/pybrain/sig.csv').readlines())
inp_o = map(lambda x: float(x.split(',')[0])-150, open('/home/martslaaf/Learn_Coding/pybrain/orig.csv').readlines())
outp_o = map(lambda x: float(x.split(',')[1]), open('/home/martslaaf/Learn_Coding/pybrain/orig.csv').readlines())
ds = []
for i in xrange(coun):
    ds.append((np.array([inp_s[i]]),  np.array([outp_s[i]])))
vs = []
for i in xrange(301):
    vs.append((np.array([inp_o[i]]),  np.array([outp_o[i]])))

n = wavelon_class_constructor()
n = n(1, 1, 19)
k = 0
track = trainer(300, ds, vs, n)
outew = []
print track
for j in vs:
    outew.append(n.forward(j[0])[0][0])
plot(outp_o, 'g')
plot(outew, 'r')
show()
