import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
import time


def dft(signal):
    N = len(signal)
    XK = []

    sum = 0
    for i in range(0, N):
        sum += signal[i]

    K0 = sum
    XK.append(K0)

    for k in range(1, N):
        sum = 0
        for n in range(0, N):
            ohmega = (-2 * np.pi * k * n) / N

            r = signal[n]

            x = round(r * np.cos(ohmega), 10)
            y = round(r * np.sin(ohmega), 10)

            # print("Oh=",ohmega,"r=", r,"x=", x,"y=", y)

            sum += complex(x, y)

        XK.append(sum)

    return np.array(XK)


def clean(ft):
    # Get rid of annoying decimal values that are almost 0 but not quite e.g -5.55111512e-17
    for i in range(0, len(ft)):
        if abs(ft[i]) < 0.01:
            ft[i] = 0

    return ft


def test():
    # Test 1 Tests the example from lectures
    dfty = dft([2, 3, 1, 4])
    result1 = [10, (1 + 1j), (-4 + 0j), (1 - 1j)]
    count = 0
    for i in range(0, len(dfty)):
        if not dfty[i] == result1[i]:
            print("Trash1")
            break
        else:
            count += 1

    if (count == 4):
        print("Good1")
    # Test 1 End

    # Test 2 Tests sin function should be 2 delta functions
    t = np.arange(0, 4 * np.pi, np.pi / 4)
    y = np.sin(t)

    dfty = dft(y)
    if (dfty[2] == -8j and dfty[-2] == 8j):
        print("Good2")
    else:
        print("Trash2")
    # End Test 2

    # Test 3 Delta Function should return a constant
    dfty = dft([10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ffty = np.fft.fft([10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Use fft to check because all results are integers

    count = 0
    for i in range(0, len(dfty)):
        if not dfty[i] == ffty[i]:
            print("Trash3")
            break
        else:
            count += 1

    if count == len(ffty):
        print("Good3")
    # End Test 3

    # Test 4 alternating between +1 and -1
    dfty = dft([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    ffty = np.fft.fft([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])

    count = 0

    clean(ffty)
    clean(dfty)

    for i in range(0, len(dfty)):
        if not dfty[i] == ffty[i]:
            print("Trash4")
            break
        else:
            count += 1

    if count == len(dfty):
        print("Good4")
    # End Test 4

    # Im not writing more tests because it took longer to write these then it did to write the dft function O.O


def main():
    start = time.perf_counter_ns()

    dft([2, 3, 1, 4])

    end = time.perf_counter_ns()

    print("Time taken for dft (seconds): ", (end - start) / 100000000)

    start = time.perf_counter_ns()

    np.fft.fft([2, 3, 1, 4])

    end = time.perf_counter_ns()

    print("Time taken for dft (seconds): ", (end - start) / 100000000)


test()