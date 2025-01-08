# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:09:30 2025

@author: RIT ECE
"""

import numpy as np
import matplotlib.pyplot as plt

bit_rate = 1e6  
rolloff_factor = 0.5
Tb = 1 / bit_rate 
num_bits = 1000
snr_db = 10  
bits = np.random.randint(0, 2, num_bits)

def root_raised_cosine(t, Tb, beta):
    if abs(t) == Tb / (4 * beta):
        return np.pi / 4 * (1 + beta)
    elif abs(t) < Tb / (4 * beta):
        return 0.5 * (1 + beta) * (1 - np.sin(np.pi * (abs(t) - Tb / (4 * beta)) / (Tb / (4 * beta))))
    elif abs(t) < Tb * (1 + 4 * beta) / (4 * beta):
        return 0.5 * (1 + beta) * (1 + np.sin(np.pi * (abs(t) - Tb * (1 + 4 * beta) / (4 * beta)) / (Tb / (4 * beta))))
    else:
        return 0


t = np.arange(-5 * Tb, 5 * Tb, Tb / 100)
pulse = np.array([root_raised_cosine(tau, Tb, rolloff_factor) for tau in t])
p=np.fft.fft(pulse)
f=np.abs(np.fft.fftshift(p))
plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)

plt.plot(f, label="Root Raised Cosine Pulse")
plt.xlim(400,600)
plt.title("Root Raised Cosine Pulse")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

s = np.zeros_like(t)
for i in range(num_bits):
    s += bits[i] * np.roll(pulse, int(i * len(t) / num_bits))
noise_power = 10 ** (-snr_db / 10)
noise = np.random.normal(0, np.sqrt(noise_power), len(s))
r = s + noise
plt.subplot(2, 2, 2)
plt.plot(t, s, label="Transmitted Signal")
plt.subplot(2, 2, 3)
plt.plot(t, r, label="Received Signal (with noise)")
plt.title("Transmitted and Received Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()


matched_filter = pulse[::-1]
y = np.convolve(r, matched_filter, mode='same')
sampling_points = y[::int(len(y) / num_bits)]
decisions = (sampling_points > 0).astype(int)

ber = np.sum(np.abs(decisions - bits)) / num_bits

plt.subplot(2, 2, 4)
plt.plot(y, label="Matched Filter Output")
plt.title("Matched Filter Output and Sampling Points")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()
print(f"Bit Error Rate (BER): {ber}")
