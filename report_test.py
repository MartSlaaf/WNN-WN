__author__ = 'martslaaf'

import numpy as np
from wavenet import wavelon_class_constructor, trainer
from wavelets import Mhat
from random import seed, uniform
from multiprocessing import Process, Queue

repeat_count = 4
def main_async_method(queue, n):
    track = trainer(50, ds, vs, n)
    queue.put({'mse': track[-1]})

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

seed()
# networks initialization
exp_1_n = []  # default set
exp_2_n = []  # default but Mhat
exp_3_n = []  # random limits
exp_4_n = []  # hidden layer x2
exp_5_n = []  # right limits for translation
exp_6_n = []  # period data (Nyqist)
exp_7_n = []  # fourier analysis
exp_8_n = []  # hidden layer /2
mini, maxi = -150, 150
e_1 = wavelon_class_constructor()
e_2 = wavelon_class_constructor(motherfunction=Mhat)
e_3 = wavelon_class_constructor(frame=(uniform(-100, 0), uniform(0, 100)), period=uniform(0, 100))
e_4 = wavelon_class_constructor()
e_5 = wavelon_class_constructor(frame=(mini, maxi))
e_6 = wavelon_class_constructor(period=120)
e_7 = wavelon_class_constructor(period=120, signal=outp_s, fa=True)
e_8 = wavelon_class_constructor()
seed()
for i in xrange(repeat_count):
    exp_1_n.append(e_1(1, 1, 19))
    exp_2_n.append(e_2(1, 1, 19))
    exp_3_n.append(e_3(1, 1, 38))
    exp_4_n.append(e_4(1, 1, 19))
    exp_5_n.append(e_5(1, 1, 19))
    exp_6_n.append(e_6(1, 1, 19))
    exp_7_n.append(e_7(1, 1, 19))
    exp_8_n.append(e_8(1, 1, 10))

list_of_experiments = [exp_1_n, exp_2_n, exp_3_n, exp_4_n, exp_5_n, exp_6_n, exp_7_n, exp_8_n]

number = 1
for experiment in list_of_experiments:
    print 'strating experiment # ', number
    process_list = []
    forests_queue = Queue(repeat_count)
    local_errors = []
    for exemplar in experiment:
        process_list.append(Process(target=main_async_method, args=(forests_queue, exemplar)))
    for proc in process_list:
        proc.start()
    for proc in process_list:
        proc.join()
    global_mse = 0
    for smth in range(forests_queue.qsize()):
        tmp = forests_queue.get()
        local_errors.append(tmp['mse'])
    print 'for experiment number ', number, ' errors are ', local_errors
    number += 1