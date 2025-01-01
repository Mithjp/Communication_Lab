'''

                            Online Python Compiler.
                Code, Compile, Run and Debug python program online.
Write your code in this editor and press "Run" button to execute it.

'''

import numpy as np
import matplotlib.pyplot as plt

fm = 4
fn = 2*fm
fs = 250*fn
t = np.arange(0,1,1/fs)
s = (np.mod(41,5)+1)*(1+np.cos(8*np.pi*t))/2
plt.plot(t,s)
plt.xlabel("t");plt.ylabel("s(t)");plt.title("plot of s(t)");

def quantize_signal(signal,no_of_bits):
   
    q_signal = []
    step_size = (max(signal)-min(signal))/2**no_of_bits
    for i in signal:
        temp = min(signal)
        if i == min(signal):
            q_signal.append(min(signal))
        else:
            while i > temp*step_size:
                temp += 1
            q_signal.append((temp-1)*step_size)
    return q_signal

q_signal_4 = quantize_signal(s,2)
q_signal_8 = quantize_signal(s,3)
q_signal_16 = quantize_signal(s,4)
q_signal_32 = quantize_signal(s,5)
q_signal_64 = quantize_signal(s,6)

plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.plot(t,q_signal_4)
plt.xlabel("t");plt.ylabel("s(t)");plt.title("s(t) quantised with L=4");
plt.subplot(2,3,2)
plt.plot(t,q_signal_8)
plt.xlabel("t");plt.ylabel("s(t)");plt.title("s(t) quantised with L=8");
plt.subplot(2,3,3)
plt.plot(t,q_signal_16)
plt.xlabel("t");plt.ylabel("s(t)");plt.title("s(t) quantised with L=16");
plt.subplot(2,3,4)
plt.plot(t,q_signal_32)
plt.xlabel("t");plt.ylabel("s(t)");plt.title("s(t) quantised with L=32");
plt.subplot(2,3,5)
plt.plot(t,q_signal_64)
plt.xlabel("t");plt.ylabel("s(t)");plt.title("s(t) quantised with L=64");

def snr_of_signal(signal,quantised_signal):
   
    noise = quantised_signal - signal
    snr = np.sum(signal**2)/np.sum(noise**2)
    return snr
snr_values = []
for i in range(2,9):
    q_signal = quantize_signal(s,i)
    snr = snr_of_signal(s,q_signal)
    snr_values.append(snr)
print(snr_values)

plt.semilogy(np.arange(2,9),snr_values)
plt.xlabel("No of bits");
plt.ylabel("SNR (log scale)");
plt.title("SNR v/s No of bits");

def encode_signal(signal,no_of_bits):
    q_signal = quantize_signal(signal,no_of_bits)
    step = (max(signal)-min(signal))/2**no_of_bits
    s1 = np.arange(min(signal),max(signal),step)
    s1 = list(s1)
    encoded_signal = []
    length = no_of_bits
    for i in q_signal:
        index = s1.index(i)
        binary = bin(index)[2:]
        code = int(length-len(binary))*'0' +binary
        encoded_signal.append(code)
    return encoded_signal
print("Encoded signal:")
print(encode_signal(s,5))