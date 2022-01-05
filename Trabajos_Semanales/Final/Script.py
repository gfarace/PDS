#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:57:02 2021

@author: gise
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.signal.windows as wind
import scipy.io as sio
from scipy.io import wavfile
from scipy.fft import fft, fftfreq, fftshift
import IPython.display as ipd

# Para la tabla
from pandas import DataFrame
from IPython.display import HTML

#%%
# Preprocesamiento:
#   2) Filtro pasabanda
#   3) Normalización
# Segmentación
#   1) FFT con ventana movil
#   2) Welch

#%%

# Pasabanda Chebyshev

def pasabanda(signal, fs): 
    nq = fs/2 #nyquist
    # Plantilla del filtro  
    fs0 = 25/nq
    fc0 = 30/nq
    fc1 = 250/nq
    fs1 = 255/nq
    # alfa_min = 30
    # alfa_max = 0.5
    alfa_min = 15
    alfa_max = 0.5
    wp = [fc0,fc1]
    ws = [fs0,fs1]
    #------------ CHEBYSHEV ------------
    sos_cheby = sig.iirdesign(wp=wp, ws=ws, gpass=alfa_max, gstop=alfa_min, analog=False, ftype='cheby1', output='sos') 
    w_cheby,h_cheby = sig.sosfreqz(sos_cheby,worN=2000, fs=fs)
    
    filtrada = sig.sosfiltfilt(sos_cheby,signal,padtype='odd',padlen=None)

    return w_cheby, h_cheby, sos_cheby, filtrada

# Normalización

def normal(signal):
    maximo = np.amax(np.abs(signal))
    return signal/maximo

# Visualizacion de graficos

def visualizacion(data):
    plt.figure()
    plt.plot(data)
    plt.title('Visualización del audio') 
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Amplitud')  
    plt.grid(True)
    return

def graf_pasabanda(h_cheby, w_cheby):
    plt.figure()
    plt.plot(w_cheby,20*np.log10(np.abs(h_cheby)), label='Chebyshev, orden {:3.0f}'.format(sos_cheby.shape[0]*2))
    plt.title('Espectro filtro pasabanda Chebyshev')
    plt.xlabel('Frecuencia [rad/sample]')
    plt.ylabel('Amplitud [dB]')
    plt.legend()
    plt.xlim(-1,270)
    plt.ylim(-25,5)
    plt.grid(True)
    return

def graf_filtrada(data_fil, data):
    plt.figure()
    # plt.plot(data, label='sin filtrar')
    plt.plot(data_fil, label='filtro Chebyshev')
    plt.plot(data, label='sin filtrar')
    plt.title('Señal filtrada con pasabanda Chebyshev')
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Amplitud')  
    plt.grid(True)
    plt.legend()
    return

def graf_fil_zoom(data_fil, data, i, f):
    plt.figure()
    # plt.plot(data, label='sin filtrar')
    plt.plot(data_fil, label='filtro Chebyshev')
    plt.plot(data, label='sin filtrar')
    plt.title('Zoom sobre señal filtrada con pasabanda Chebyshev')
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Amplitud')  
    plt.grid(True)
    plt.legend()
    plt.xlim(i,f)
    return

def segmentacion(data_norm, fs, Npad, muestras):
    inicio = 0
    cantidad = np.arange(inicio,data_norm.size, muestras)
    data_norm = data_norm.reshape(data_norm.size,1)
    matrix = [ (data_norm[int(i):int(i+muestras)]) for i in cantidad ]
    matrix = np.hstack(matrix[0:len(matrix)-1])
    return matrix

def graf_fft(fft_latidos, fs, Npad):
    T = 1.0 / fs
    xf = fftfreq(Npad, T)[:Npad//2]
    plt.figure() 
    for i in range(0,fft_latidos.shape[1]):
        plt.plot(xf, np.abs(fft_latidos[0:Npad//2, i])/np.amax(np.abs(fft_latidos[0:Npad//2, i])))
    plt.title('FFT con ventanas de 500ms')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')  
    plt.axvline(x=50, color='r', lw=4)
    plt.axvline(x=100, color='r', lw=4)
    plt.axvline(x=200, color='r', lw=4)
    plt.xlim(10,300)
    plt.grid()
    plt.show()
    return

def graf_fft_2(fft_latidos, fs, Npad):
    T = 1.0 / fs
    xf = fftfreq(Npad, T)[:Npad//2]
    plt.figure() 
    for i in range(0,4):
        plt.plot(xf, np.abs(fft_latidos[0:Npad//2, i])/np.amax(np.abs(fft_latidos[0:Npad//2, i])))
    plt.title('FFT con ventanas de 500ms, observando menos muestras')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')  
    plt.axvline(x=50, color='r', lw=4)
    plt.axvline(x=100, color='r', lw=4)
    plt.axvline(x=200, color='r', lw=4)
    plt.xlim(10,300)
    plt.grid()
    plt.show()
    return

def graf_welch(Pw_original, fw, corte_energia):
    norm = np.amax(Pw_original)
    Pw_original = np.mean(Pw_original,axis=1)
    Pot = np.cumsum(Pw_original)/np.sum(Pw_original)
    corte = np.where(Pot >corte_energia)[0][0]

    plt.figure()
    plt.plot(fw,Pw_original/norm, 'k')
    plt.fill_between(fw, 0, Pw_original/norm, where = fw < fw[corte], color='blue')
    plt.title('Ancho de banda donde se concentra el {:3.0f}% de la energia'.format(corte_energia*100))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [$V^{2}/Hz$]')
    plt.xlim(0,250)
    
    plt.annotate(   "BW_n = {:3.1f} Hz".format(fw[corte]),
                    xy=(fw[corte], Pw_original[corte]/norm),
                    xytext=(-20,20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle='->')
    )
    return

    
    
#%%
plt.close('all')

# Visualización de la señal
fs, data = wavfile.read("fourth_heart_sound_es.wav")

data = data[:,0]
visualizacion(data)

# ------------------- PREPROCESAMIENTO ------------------- #

# Filtro pasabanda
w_cheby, h_cheby, sos_cheby, data_fil = pasabanda(data, fs)

# Grafico el pasabanda
graf_pasabanda(h_cheby, w_cheby)

# Grafico señal filtrada
graf_filtrada(data_fil, data)

i=23000
f=35000
graf_fil_zoom(data_fil, data, i, f)

# Descargo audio filtrado
#wavfile.write("filtrado1.wav", fs, data_fil.astype(np.int16))

# Normalizo
data_norm = normal(data_fil)
# plt.figure(4)
# plt.plot(data_norm)
# plt.title('Señal normalizada')
# plt.xlabel('Tiempo (ms)')
# plt.ylabel('Amplitud')  
# plt.grid(True)

#%%
plt.close('all')
# --------------------- SEGMENTACION --------------------- #
# FFT con ventana movil: hago ventanas cada 500ms
Npad = fs*2
muestras = fs/2 # Hago muestras de 500ms

data_seg = segmentacion(data_norm, fs, Npad, muestras)
fft_latidos =  fft(data_seg, axis = 0, n=Npad)

graf_fft(fft_latidos, fs, Npad)

#%%
plt.close('all')

# -------- WELCH ---------
fw, Pw_original = sig.welch(data_seg, fs=fs, axis=0, nperseg=data_seg[:,0].size/2,window='bartlett')
corte_energia = 0.9
graf_welch(Pw_original, fw, corte_energia)
