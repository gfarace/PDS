#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28/9/2021

@author: Gisela Farace

Descripción: Tarea semanal 7
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
#%%

plt.close('all')

#Senoidal
#x(k)=a0⋅sen(Ω1⋅k)
#a0=2
#Ω1=Ω0+fr⋅2πN
#Ω0=π2

a0 = 2
M = 200
dc = 0
ph = 0
N = 1000  # cantidad de muestras
fs = 1000 # frecuencia de muestreo (Hz)
df = fs/N # resolución espectral

# distribución uniforme
fr = np.random.uniform(low=-2, high=2, size=M)

ff = (np.pi/2 + fr*(2*np.pi/N))*(fs/(2*np.pi))
tt = np.linspace(0, (N-1), N) * (1/fs)

#np.outer(a,b): Compute the outer product of two vectors.
x = a0*np.sin(2*np.pi*np.outer(tt, ff))

# Multiplico la señal por cada ventanas
s_rec = x*np.array(wind.boxcar(N)).reshape(N,1)
s_bar = x*np.array(wind.bartlett(N)).reshape(N,1)
s_han = x*np.array(wind.hann(N)).reshape(N,1)
s_bm = x*np.array(wind.blackman(N)).reshape(N,1)
s_ft = x*np.array(wind.flattop(N)).reshape(N,1)

# Transformada
fft_rec = np.fft.fft(s_rec, n = N, axis = 0)*(1/N)
fft_bar = np.fft.fft(s_bar, n = N, axis = 0)*(1/N)
fft_han = np.fft.fft(s_han, n = N, axis = 0)*(1/N)
fft_bm = np.fft.fft(s_bm, n = N, axis = 0)*(1/N)
fft_ft = np.fft.fft(s_ft, n = N, axis = 0)*(1/N)

frec = np.fft.fftfreq(N, d=1/fs)

#Estimadores (â)
rec_h = np.abs(fft_rec[frec == 250,:]).flatten()
bar_h = np.abs(fft_bar[frec == 250,:]).flatten()
han_h = np.abs(fft_han[frec == 250,:]).flatten()
bm_h = np.abs(fft_bm[frec == 250,:]).flatten()
ft_h = np.abs(fft_ft[frec == 250,:]).flatten()

# Histogramas

plt.hist(rec_h, bins=20)
plt.hist(bar_h, bins=20)
plt.hist(han_h, bins=20)
plt.hist(bm_h, bins=20)
plt.hist(ft_h, bins=20)

plt.title('Histograma para distintas ventanas')
plt.legend(['Rectangular','Bartlett','Hann','Blackman','Flattop'])
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('$|X(\Omega)|$')

# Calculo de sesgo y varianza

# media muestral: mu_a = 1/M.SUM{â_j} de 0 a M-1
E_rec = (1/M)*sum(rec_h)
E_bar = (1/M)*sum(bar_h)
E_han = (1/M)*sum(han_h)
E_bm = (1/M)*sum(bm_h)
E_ft = (1/M)*sum(ft_h)

#Sesgo: s = mu - a0
s_rec = E_rec - a0
s_bar = E_bar - a0
s_han = E_han - a0
s_bm = E_bm - a0
s_ft = E_ft - a0

# Varianza:
v_rec = np.var(rec_h)
v_bar = np.var(bar_h)
v_han = np.var(han_h)
v_bm = np.var(bm_h)
v_ft = np.var(ft_h)

data = [[s_rec, v_rec],
        [s_bar, v_bar],
        [s_han, v_han],
        [s_bm, v_bm],
        [s_ft, v_ft]]

df = DataFrame(data,columns=['$s_a$', '$v_a$'],
                index=[  
                        'Rectangular',
                        'Bartlett',
                        'Hann',
                        'Blackman',
                        'Flat-top'
                      ])
HTML(df.to_html())

