__author__ = 'martslaaf'
from numpy import exp, cos, sin

class Morlet():
    """ Morlet wavelet motherfunction.
    """
    @staticmethod
    def function(x):
        return 0.75112554446494 * cos(x * 5.336446256636997) * exp((-x ** 2) / 2)

    @staticmethod
    def derivative(x):
        return -4.008341100024355 * exp((-x ** 2) / 2) * sin(5.336446256636997 * x) - 0.75112554446494 * x * exp((-x ** 2) / 2) * cos(5.336446256636997 * x)

    @staticmethod
    def from_freq(freq):
        return 0.8458 / freq + 0.0005407

class Mhat():
    """ Mexican hat wavelet motherfunction.
    """
    @staticmethod
    def function(x):
        return (1 - x ** 2) * exp((-x ** 2) / 2)

    @staticmethod
    def derivative(x):
        return -x * (1 - x ** 2) * exp((-x ** 2) / 2) - 2 * x * exp((-x ** 2) / 2)

    @staticmethod
    def from_freq(freq):
        return abs(0.2282 / freq - 0.001325)