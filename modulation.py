# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
# Get started with interactive Python!
# Supports Python Modules: builtins, math,pandas, scipy
# matplotlib.pyplot, numpy, operator, processing, pygal, random,
# re, string, time, turtle, urllib.request
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def butterFilt(st, fs, fm):
    Fp = (fm * 2) / fs
    Fs = (fm * 4) / fs
    Fc = (Fs + Fp) / 2

    b, a = sp.signal.butter(5, Fc, btype="low", analog=False)

    y = sp.signal.lfilter(b, a, np.abs(st))

    return y


def main():
    fc = 350000

    Am = 1
    fm = 5

    modIndex = 0.75

    Ac = Am / modIndex

    samples = 5000
    range = 1
    fs = samples / range

    t = np.linspace(0, range, samples)

    carrier = np.cos(2 * np.pi * fc * t)

    message = Am * np.cos(2 * np.pi * fm * t)

    st = Ac * carrier + message * carrier

    y = butterFilt(st, fs, fm)

    plt.subplot(4, 1, 1)
    plt.title('Amplitude Modulation')
    plt.plot(t, message, 'g')
    plt.ylabel('Amplitude')
    plt.xlabel('Message Signal')

    plt.subplot(4, 1, 2)
    plt.plot(t, Ac * carrier, 'r')
    plt.ylabel('Amplitude')
    plt.xlabel('Carrier Signal')

    plt.subplot(4, 1, 3)
    plt.plot(t, st)
    plt.ylabel('Amplitude')
    plt.xlabel('AM Signal')

    plt.subplot(4, 1, 4)
    plt.plot(t, y, 'g')
    plt.ylabel('Amplitude')
    plt.xlabel('Demodulated Signal')

    plt.subplots_adjust(hspace=1)
    fig = plt.gcf()
    fig.set_size_inches(28, 12)

    plt.show()


main()