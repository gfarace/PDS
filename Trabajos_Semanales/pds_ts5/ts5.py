#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28/9/2021

@author: Gisela Farace

Descripción: Tarea semanal 5
------------
"""

# Importación de módulos para Jupyter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig


def senoidal(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs # tiempo de muestreo
    df = fs/nn # resolución espectral 
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    
    # grilla de sampleo frecuencial
    sen = vmax*np.sin(2*np.pi*ff*tt + ph)+dc
    
    return tt, sen

def fft(xx, N):
    ft = np.fft.fft(xx, axis=0)
    ft = ft/N 
    return ft
#%%
# x = x'.w
# x' es una senoidal con fo=k0∗fS/N=k0.Δf
# X = (X'*W).1/2pi
# W(k) = e^fase.(sen(pi.k)/sen(pi.k/N)), donde fase=-j.pi.k(1-1/N)
# k0={N/4,N/4+0.025,N/4+0.5}

plt.close('all')

# datos de la senoidal
vmax = 1
dc = 0
ph = 0
N = 1000  # cantidad de muestras
fs = 1000 # frecuencia de muestreo (Hz)
df = fs/N # resolucion espectral

freq=[(N/4)*(fs/N),(N/4)*(fs/N)+0.25,(N/4)*(fs/N)+0.5]
j = 0
pot = [0,0,0]

for ff in freq:
    tt,xx = senoidal(vmax, dc, ff, ph, N, fs)
    xx_nor = xx/np.sqrt(np.var(xx,axis=0)) #Normalizo la señal
    fft_xx = fft(xx_nor, N)
    pot[j]=np.sum(np.abs(fft_xx)**2) #Potencia
    df=fs/N
    f = np.linspace(0, (N-1), N)*df
    bfrec = f <= fs/2
    #busco graficar en db para que se vea mejor
    plt.plot(f[bfrec], 10*np.log10(2*np.abs(fft_xx[bfrec])**2),'x:')  
    j+=1

plt.legend(['Fs={:3.2f} Hz Pot={:1.2f}'.format(freq[0], pot[0]),'Fs={:3.2f} Hz Pot={:1.2f}'.format(freq[1], pot[1]),'Fs={:3.2f} Hz Pot={:1.2f}'.format(freq[2], pot[2])])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.ylim(-100,5)

#%% Zero Padding

plt.close('all')

# datos de la senoidal
vmax = 1
dc = 0
ph = 0
N = 1000  # cantidad de muestras
fs = 1000 # frecuencia de muestreo (Hz)
Npad = 9*N
df = fs/Npad # resolucion espectral
freq=[(Npad/4)*(fs/Npad),(Npad/4)*(fs/Npad)+0.25,(Npad/4)*(fs/Npad)+0.5]
j = 0
pot = [0,0,0]
pot_pad = [0,0,0]

for ff in freq:
    tt,xx = senoidal(vmax, dc, ff, ph, N, fs)
    xx_nor = xx/np.sqrt(np.var(xx,axis=0)) #Normalizo la señal
    #agrego los ceros
    xx_pad = xx_nor.copy() 
    xx_pad.resize(Npad)
    xx_pad = xx_pad/np.sqrt(np.var(xx_pad,axis=0)) #Normalizo la señal
    fft_xx = fft(xx_nor, N)
    fft_xx_pad = fft(xx_pad, Npad)
    pot[j] = np.sum(np.abs(fft_xx)**2) #Potencia
    pot_pad[j] = np.sum(np.abs(fft_xx_pad)**2) #Potencia padding
    f1 = np.linspace(0, (N-1), N)*fs/N
    bfrec1 = f1 <= fs/2
    f = np.linspace(0, (Npad-1), Npad)*df
    bfrec = f <= fs/2
    #busco graficar en db para que se vea mejor
    plt.plot(f1[bfrec1], 10*np.log10(2*np.abs(fft_xx[bfrec1])**2),'x:')  
    plt.plot(f[bfrec], 10*np.log10(2*np.abs(fft_xx_pad[bfrec])**2),'x:')  
    j+=1

plt.legend(['Fs={:3.2f} Hz Pot={:1.2f}'.format(freq[0], pot[0]),'Fs_pad={:3.2f} Hz Pot={:1.2f}'.format(freq[0], pot_pad[0]),'Fs={:3.2f} Hz Pot={:1.2f}'.format(freq[1], pot[1]),'Fs_pad={:3.2f} Hz Pot={:1.2f}'.format(freq[1], pot_pad[1]),'Fs={:3.2f} Hz Pot={:1.2f}'.format(freq[2], pot[2]),'Fs_pad={:3.2f} Hz Pot={:1.2f}'.format(freq[2], pot_pad[2])])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlim(240,260)
plt.ylim(-50,5)

#%% Zero Padding

plt.close('all')

# datos de la senoidal
vmax = 1
dc = 0
ph = 0
N = 1000  # cantidad de muestras
fs = 1000 # frecuencia de muestreo (Hz)
Npad = 9*N
df = fs/Npad # resolucion espectral
freq=[(Npad/4)*(fs/Npad),(Npad/4)*(fs/Npad)+0.25,(Npad/4)*(fs/Npad)+0.5]
j = 0
pot = [0,0,0]
pot_pad = [0,0,0]

for ff in freq:
    tt,xx = senoidal(vmax, dc, ff, ph, N, fs)
    xx_nor = xx/np.sqrt(np.var(xx,axis=0)) #Normalizo la señal
    #agrego los ceros
    xx_pad = xx_nor.copy() 
    xx_pad.resize(Npad)
    xx_pad = (Npad/N)*xx_pad
    fft_xx = fft(xx_nor, N)
    fft_xx_pad = fft(xx_pad, Npad)
    pot[j] = np.sum(np.abs(fft_xx)**2) #Potencia
    pot_pad[j] = np.sum(np.abs(fft_xx_pad)**2) #Potencia padding
    f1 = np.linspace(0, (N-1), N)*fs/N
    bfrec1 = f1 <= fs/2
    f = np.linspace(0, (Npad-1), Npad)*df
    bfrec = f <= fs/2
    #busco graficar en db para que se vea mejor
    plt.plot(f1[bfrec1], 10*np.log10(2*np.abs(fft_xx[bfrec1])**2),'x:')  
    plt.plot(f[bfrec], 10*np.log10(2*np.abs(fft_xx_pad[bfrec])**2),'o:',mfc='none')  
    j+=1

plt.legend(['Fs={:3.2f} Hz Pot={:1.2f}'.format(freq[0], pot[0]),'Fs_pad={:3.2f} Hz Pot={:1.2f}'.format(freq[0], pot_pad[0]),'Fs={:3.2f} Hz Pot={:1.2f}'.format(freq[1], pot[1]),'Fs_pad={:3.2f} Hz Pot={:1.2f}'.format(freq[1], pot_pad[1]),'Fs={:3.2f} Hz Pot={:1.2f}'.format(freq[2], pot[2]),'Fs_pad={:3.2f} Hz Pot={:1.2f}'.format(freq[2], pot_pad[2])])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlim(245,255)
plt.ylim(-30,5)
