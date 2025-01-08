# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:59:54 2025

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


pulse = [root_raised_cosine(tau, Tb, rolloff_factor) for tau in t]


s = np.zeros_like(t)
for i in range(num_bits):
    s += bits[i] * np.roll(pulse, int(i * len(t) / num_bits))


noise_power = 10**(-snr_db / 10)
noise = np.random.normal(0, np.sqrt(noise_power), len(s))
r = s + noise

matched_filter = pulse[::-1]


y = np.convolve(r, matched_filter, mode='same')


samples = y[::int(len(y) / num_bits)]


decisions = (samples > 0).astype(int)


ber = np.sum(np.abs(decisions - bits)) / num_bits


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, s, label='Transmitted Signal')
plt.plot(t, r, label='Received Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(samples, label='Matched Filter Output')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.show()

print(f'BER: {ber}')
