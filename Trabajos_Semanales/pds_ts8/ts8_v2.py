#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/10/2021

@author: Gisela Farace

Descripción: Tarea semanal 8
------------
"""

# Importación de módulos para Jupyter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.signal.windows as wind
# Para la tabla
from pandas import DataFrame
from IPython.display import HTML

from spectrum import pcorrelogram as BlackmanTukey


#%%
plt.close('all')
#Senoidal
#x(k)=a1⋅sen(Ω1⋅k)+n(k)
#Ω1=Ω0+(fr.2π)/N
#Ω0 = pi/2

N = 1000  # cantidad de muestras
fs = 1000 # frecuencia de muestreo (Hz)
R = 200 # realizaciones
Npad = N*10

SNR = np.array([3, 10], dtype=float) #En dB
# SNR = 10.log(Ps/Pn)
# Pn = Ps/[10^(SNR/10)]
# Ps = A^2/2 --> A = sqrt(Ps*2)
pot_sen = 1 #potencia unitaria
a1 = np.sqrt(pot_sen*2) # amplitud senoidal
pot_ruido = pot_sen/(10**(SNR/20))

ts = 1/fs # tiempo de muestreo (Hz)
df = fs/N # resolución espectral
tt = np.linspace(0, (N-1), N) * (1/fs)

frec = np.linspace(0, (N-1), N)*df
frec_p = np.linspace(0, (Npad-1), Npad)*fs/N

# Variables aleatorias
fr = np.random.uniform(low=-1/2, high=1/2, size=(N,R)) # distribución uniforme
n_3db = np.random.normal(0, np.sqrt(pot_ruido[0]), size=(N,R)) # ruido con SNR 3dB
n_10db = np.random.normal(0, np.sqrt(pot_ruido[1]), size=(N,R)) # ruido con SNR 10dB

ff = (np.pi/2 + fr*(2*np.pi/N))*(fs/(2*np.pi))
x = a1*np.sin(2*np.pi*ff*tt.reshape(N,1))

#################### Para SNR=3dB ####################
x1 = x + n_3db
#----------------- Periodograma -----------------#
fft_x1_p = np.fft.fft(x1, axis = 0, n=Npad)
x1_per = (1/N)*(np.abs(fft_x1_p)**2)

plt.figure(1)
plt.plot(frec_p,x1_per)
plt.title('Periodograma para SNR={:1.0f}dB'.format(SNR[0]))
plt.xlabel('Frecuencia')
plt.ylabel('PSD')
plt.xlim(2450,2550)

#----------------- Welch -----------------#
fw_x1_w, x1_welch = sig.welch(x1, fs, nfft=Npad, axis=0)
x1_picos_w = np.argmax(x1_welch, axis=0)/(Npad/N)

plt.figure(2)
plt.plot(fw_x1_w,x1_welch)
plt.title('Welch para SNR={:1.0f}dB'.format(SNR[0]))
plt.xlabel('Frecuencia')
plt.ylabel('PSD')
plt.xlim(220,280)

#----------------- Blackman Tukey -----------------#
# la funcion del spectrum no tiene para indicarle axis
x1_picos_b = np.zeros(R)
x1_black = np.zeros(R)
plt.figure(3)
for i in range(0,R):
    x1black = BlackmanTukey(x1[:,i],lag=15, NFFT=Npad, sampling=fs)
    x1_black = x1black.psd
    x1_picos_b[i] = np.argmax(x1_black, axis=0)/(Npad/N)
    x1black.plot(norm=True)
plt.title('Blackman Tukey para SNR={:1.0f}dB'.format(SNR[0]))
plt.xlabel('Frecuencia')
plt.ylabel('PSD')
plt.ylim(-9,1)
    
#%%  
plt.close('all')  
#################### Para SNR=10dB ####################
x2 = x + n_10db
#----------------- Periodograma -----------------#
fft_x2_p = np.fft.fft(x2, axis = 0, n=Npad)
x2_per = (1/N)*(np.abs(fft_x2_p)**2)

plt.figure(1)
plt.plot(frec_p,x2_per)
plt.title('Periodograma para SNR={:1.0f}dB'.format(SNR[1]))
plt.xlabel('Frecuencia')
plt.ylabel('PSD')
plt.xlim(2450,2550)

#----------------- Welch -----------------#
fw_x2_w, x2_welch = sig.welch(x2, fs, nfft=Npad, axis=0)
x2_picos_w = np.argmax(x2_welch, axis=0)/(Npad/N)

plt.figure(2)
plt.plot(fw_x2_w,x2_welch)
plt.title('Welch para SNR={:1.0f}dB'.format(SNR[1]))
plt.xlabel('Frecuencia')
plt.ylabel('PSD')
plt.xlim(220,280)

#----------------- Blackman Tukey -----------------#
# la funcion del spectrum no tiene para indicarle axis
x2_picos_b = np.zeros(R)
x2_black = np.zeros(R)
plt.figure(3)
for i in range(0,R):
    x2black = BlackmanTukey(x2[:,i],lag=15, NFFT=Npad, sampling=fs)
    x2_black = x2black.psd
    x2_picos_b[i] = np.argmax(x2_black, axis=0)/(Npad/N)
    x2black.plot(norm=True)
plt.title('Blackman Tukey para SNR={:1.0f}dB'.format(SNR[1]))
plt.xlabel('Frecuencia')
plt.ylabel('PSD')
plt.ylim(-11,1)

#%%
err_3_welch = x1_picos_w - ff
err_3_black = x1_picos_b - ff
err_10_welch = x2_picos_w - ff
err_10_black = x2_picos_b - ff


data = [[np.mean(err_3_welch), np.mean(err_10_welch)],
        [np.var(err_3_welch), np.var(err_10_welch)],
        [np.mean(err_3_black), np.mean(err_10_black)],
        [np.var(err_3_black), np.var(err_10_black)]]

df = DataFrame(data,columns=['3dB', '10dB' ],
                index=[  
                        'Welch sesgo',
                        'Welch varianza',
                        'Blackamn sesgo',
                        'Blackman varianza',
                      ])
HTML(df.to_html())


