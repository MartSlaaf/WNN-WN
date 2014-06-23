__author__ = 'martslaaf'
import numpy as np
from random import shuffle

from wavelets import Morlet


etta = 0.01
def from_signal_freq(signal, nyq_freq):
    fourier = abs(np.fft.fft(signal))
    positive = fourier[:int(fourier.shape[-1]/2)]
    needed = sum(list(positive)) * 0.95
    maximum = np.argmax(positive)
    right_point = maximum
    left_point = maximum
    while needed > 0:
        current = np.argmax(positive)
        if current > right_point:
            right_point = current
        if current < left_point:
            left_point = current
        needed -= positive[current]
        positive[current] = 0
    step = nyq_freq / positive.shape[-1]
    min_freq = left_point * step if left_point > 0 else 0.5 * step
    max_freq = right_point * step
    return min_freq, max_freq


def wavelon_class_constructor(motherfunction=None, period=None, frame=None, signal=None, fa=False):
    Mtf = motherfunction if motherfunction else Morlet

    if period:
        freq_Nyquist = 1.0 / (2 * period)
        if signal and fa:
            min_freq, max_freq = from_signal_freq(signal, freq_Nyquist)
        elif signal:
            min_freq, max_freq = freq_Nyquist / len(signal), freq_Nyquist
        else:
            min_freq, max_freq = 0.2e-300, freq_Nyquist
    else:
        min_freq, max_freq = 0.2e-300, 1

    min_dela, max_dela = Mtf.from_freq(max_freq), Mtf.from_freq(max_freq)

    if frame:
        min_trans, max_trans = frame
    else:
        min_trans, max_trans = -1, 1

    class Wavelon():

        def __init__(self, indim, outdim, hiddim):
            self.indim = indim
            self.outdim = outdim
            self.hiddim = hiddim
            self.inconnections = np.random.random_sample((indim, hiddim))
            self.outconnections = np.random.random_sample((hiddim, outdim))
            np.random.seed()
            self.summer = np.random.random_sample((1, outdim))
            self.translations = np.random.random_sample((1, hiddim)) * (max_trans - min_trans) + min_trans
            self.dilations = np.random.random_sample((1, hiddim)) * (max_dela - min_dela) + min_dela
            self._mother = Mtf
            self.wavemodeon = True
            self.old_data = {'summer': 0, 'inconnections': 0, 'outconnections': 0, 'translations': 0, 'dilations': 0}

        def forward(self, input):
            U = np.reshape(input, (1, self.indim))
            a = np.dot(self._mother.function((np.dot(U, self.inconnections) - self.translations)/self.dilations), self.outconnections) + self.summer
            return a

        def backup(self, delta_Chi, delta_M, delta_Omega, delta_T=None, delta_Lambda=None):
            def step(x, y, o):
                return x + y * etta + etta * (x - o)
            new = {}
            new['summer'] = step(self.summer, delta_Chi, self.old_data['summer'])
            if self.wavemodeon:
                new['dilations'] = step(self.dilations, delta_Lambda, self.old_data['dilations'])
                new['translations'] = step(self.translations, delta_T, self.old_data['translations'])
            new['inconnections'] = step(self.inconnections, delta_Omega, self.old_data['inconnections'])
            new['outconnections'] = step(self.outconnections, delta_M, self.old_data['outconnections'])
            self.old_data['summer'] = self.summer
            if self.wavemodeon:
                self.old_data['dilations'] = self.dilations
                self.old_data['translations'] = self.translations
            self.old_data['inconnections'] = self.inconnections
            self.old_data['outconnections'] = self.outconnections
            self.summer = new['summer']
            if self.wavemodeon:
                self.dilations = new['dilations']
                self.translations = new['translations']
            self.inconnections = new['inconnections']
            self.outconnections = new['outconnections']

        def backward(self, error, input):
            U = np.reshape(input, (1, self.indim))
            Err = np.reshape(error, (1, self.outdim))
            Z = self._mother.function((np.dot(U, self.inconnections) - self.translations)/self.dilations)
            Zs = self._mother.derivative((np.dot(U, self.inconnections) - self.translations)/self.dilations)
            delta_Chi = Err
            delta_M = np.dot(Z.transpose(), Err)
            # print U.shape
            delta_Omega = np.dot(U.transpose(), (np.dot(self.outconnections, Err.transpose()).transpose()*(Zs/self.dilations)))
            # print Z.shape, Zs.shape, delta_Chi.shape, delta_M.shape, delta_Omega.shape
            if self.wavemodeon:
                delta_T = np.dot(Err, self.outconnections.transpose())*(Zs/self.dilations)
                delta_Lambda = Zs*((np.dot(U, self.inconnections) - self.translations)/(self.dilations*self.dilations))
                self.backup(delta_Chi, delta_M, delta_Omega, delta_T, delta_Lambda)
            else:
                self.backup(delta_Chi, delta_M, delta_Omega)

    return Wavelon


def trainer(epochs, training, validation, net):
    track = []
    for i in xrange(epochs):
        localtrain = training[:]
        shuffle(localtrain)
        for element in localtrain:
            net.backward(element[1] - net.forward(element[0]), element[0])
        local_mse = 0.0
        for element in validation:
            local_mse += 0.5 * sum((element[1] - net.forward(element[0])) ** 2)
        track.append(local_mse / (len(validation) * net.outdim))
    return track
