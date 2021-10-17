#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28/9/2021

@author: Gisela Farace

Descripción: Tarea semanal 6
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

#%% Ventanas
plt.close('all')

N = 1000
fs = 1000
Npad = 10*N

rec = wind.boxcar(N)
bar = wind.bartlett(N)
han = wind.hann(N)
bm = wind.blackman(N)
ft = wind.flattop(N)

tt = np.linspace(0, (N-1), N)

plt.figure(1)
plt.plot(tt, rec)
plt.plot(tt, bar)
plt.plot(tt, han)
plt.plot(tt, bm)
plt.plot(tt, ft)

plt.title('Ventanas')
plt.legend(['Rectangular','Bartlett','Hann','Blackman','Flattop'])
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

# Calculo la transformada de cada una, normalizando para que estén en 0dB

fft_rec = np.fft.fft(rec, axis=0, n=Npad)
fft_bar = np.fft.fft(bar, axis=0, n=Npad)
fft_han = np.fft.fft(han, axis=0, n=Npad)
fft_bm = np.fft.fft(bm, axis=0, n=Npad)
fft_ft = np.fft.fft(ft, axis=0, n=Npad)

f = np.linspace(0, (Npad-1), Npad)*fs/N
bfrec = f <= fs/2

fft_rec_db = 20*np.log10(np.abs(1/Npad*fft_rec[bfrec])/np.abs(1/Npad*fft_rec[0]))
fft_bar_db = 20*np.log10(np.abs(1/Npad*fft_bar[bfrec])/np.abs(1/Npad*fft_bar[0]))
fft_han_db = 20*np.log10(np.abs(1/Npad*fft_han[bfrec])/np.abs(1/Npad*fft_han[0]))
fft_bm_db = 20*np.log10(np.abs(1/Npad*fft_bm[bfrec])/np.abs(1/Npad*fft_bm[0]))
fft_ft_db = 20*np.log10(np.abs(1/Npad*fft_ft[bfrec])/np.abs(1/Npad*fft_ft[0]))

plt.figure(2)
plt.plot(f[bfrec], fft_rec_db,'x:')  
plt.plot(f[bfrec], fft_bar_db,'x:')  
plt.plot(f[bfrec], fft_han_db,'x:')  
plt.plot(f[bfrec], fft_bm_db,'x:')  
plt.plot(f[bfrec], fft_ft_db,'x:')  

plt.legend(['Rectangular','Bartlett','Hann','Blackman','Flattop'])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlim(0,125)
plt.ylim(-150,5)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Busco las frecuencias de los cruces por cero

def zero_1(fft_ven_db, N, Npad):
    zero1 = 0;
    frec1 = 0;
    for i in range (0,int(N/50)):
        if fft_ven_db[i] < zero1:
            zero1 = fft_ven_db[i]
            frec1 = i
    frec1 = frec1*fs/Npad  
    return frec1;

def zero_2(fft_ven_db, Npad):
    # np.where out: ndarray An array with elements from x where condition is True, and elements from y elsewhere
    frec2 = np.where(fft_rec_db == find_nearest(fft_rec_db,-3))[0][0]* fs/Npad
    return frec2

def Wmax(fft_ven_db, N):
    W_max = -N
    for i in range (int(N/5),int(N/2)):
        if fft_ven_db[i] > W_max:
            W_max = fft_ven_db[i] 
    return W_max

fft_vent = [fft_rec_db,fft_bar_db,fft_han_db,fft_bm_db,fft_ft_db]
z1 = [0,0,0,0,0]
z2 = [0,0,0,0,0]
Wm = [0,0,0,0,0]

for i in range(0,5):
    z1[i]= zero_1(fft_vent[i], N, Npad)
    z2[i] = zero_2(fft_vent[i], Npad)
    Wm[i] = Wmax(fft_vent[i], N)
    
#Tabla

data = [[z1[0], z2[0], Wm[0]],
        [z1[1], z2[1], Wm[1]],
        [z1[2], z2[2], Wm[2]],
        [z1[3], z2[3], Wm[3]],
        [z1[4], z2[4], Wm[4]]]

df = DataFrame(data,columns=['$\Omega_0$', '$\Omega_1$', '$W_2$' ],
                index=[  
                        'Rectangular',
                        'Bartlett',
                        'Hann',
                        'Blackman',
                        'Flat-top'
                      ])
HTML(df.to_html())

#%% 

plt.close('all')

plt.figure(2)
plt.plot(f[bfrec], fft_rec_db,'x:')  
plt.plot(f[bfrec], fft_bar_db,'x:')  
plt.plot(f[bfrec], fft_han_db,'x:')  
plt.plot(f[bfrec], fft_bm_db,'x:')  
plt.plot(f[bfrec], fft_ft_db,'x:')  

plt.legend(['Rectangular','Bartlett','Hann','Blackman','Flattop'])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlim(0,125)
plt.ylim(-150,5)

plt.scatter(ff[min_[i-1]], fftx_db[min_[i-1]], color='green')
plt.scatter(ff[fftx_db == p_3db][0], p_3db, color='y')
plt.scatter(ff[max_[i-1]], fftx_db[max_[i-1]], color='r')





#%%

plt.close('all')

vmax = 1
dc = 0
ph = 0
N = 1000  # cantidad de muestras
fs = 1000 # frecuencia de muestreo (Hz)

Npad = 10*N

# Ventanas
rec = wind.boxcar(N)
bar = wind.bartlett(N)
han = wind.hann(N)
bm = wind.blackman(N)
ft = wind.flattop(N)

# 1 rad/s = 1/(2π) Hz.

K=0

ff1 = (np.pi/2)*(fs/(2*np.pi))+K
ff2 = (np.pi/2+10*2*np.pi/N)*(fs/(2*np.pi))

vmax2 = 10**(-40/20)

tt,xx1 = senoidal(vmax, dc, ff1, ph, N, fs)
tt,xx2 = senoidal(vmax2, dc, ff2, ph, N, fs)

# x = xx1 + xx2
# fft_xx = np.fft.fft(x, axis=0)
# maximo = (np.abs(fft_xx)).argmax()
# fft_nor = np.abs(fft_xx)/np.abs(fft_xx[maximo])
# f = np.linspace(0, (N-1), N)*fs/N
# bfrec = f <= fs/2

# plt.plot(f[bfrec], 10*np.log10(2*np.abs(fft_nor[bfrec])**2),'x:')

f = np.linspace(0, (Npad-1), Npad)*fs/N
bfrec = f <= fs/2


x = xx1 + xx2
x_rec = x*rec
fft_xx = np.fft.fft(x_rec, axis=0, n=Npad)
maximo = (np.abs(fft_xx)).argmax()
fft_nor_rec = np.abs(fft_xx)/np.abs(fft_xx[maximo])

plt.plot(f[bfrec], 10*np.log10(2*np.abs(fft_nor_rec[bfrec])**2),'x:')

x_bar = x*bar
fft_xx = np.fft.fft(x_bar, axis=0, n=Npad)
maximo = (np.abs(fft_xx)).argmax()
fft_nor_bar = np.abs(fft_xx)/np.abs(fft_xx[maximo])

plt.plot(f[bfrec], 10*np.log10(2*np.abs(fft_nor_bar[bfrec])**2),'x:')

x_han = x*han
fft_xx = np.fft.fft(x_han, axis=0, n=Npad)
maximo = (np.abs(fft_xx)).argmax()
fft_nor_han = np.abs(fft_xx)/np.abs(fft_xx[maximo])

plt.plot(f[bfrec], 10*np.log10(2*np.abs(fft_nor_han[bfrec])**2),'x:')

x_bm = x*bm
fft_xx = np.fft.fft(x_bm, axis=0, n=Npad)
maximo = (np.abs(fft_xx)).argmax()
fft_nor_bm = np.abs(fft_xx)/np.abs(fft_xx[maximo])

plt.plot(f[bfrec], 10*np.log10(2*np.abs(fft_nor_bm[bfrec])**2),'x:')

x_ft = x*ft
fft_xx = np.fft.fft(x_ft, axis=0, n=Npad)
maximo = (np.abs(fft_xx)).argmax()
fft_nor_ft = np.abs(fft_xx)/np.abs(fft_xx[maximo])

plt.plot(f[bfrec], 10*np.log10(2*np.abs(fft_nor_ft[bfrec])**2),'x:')


plt.legend(['Rectangular','Bartlett','Hann','Blackman','Flattop'])






