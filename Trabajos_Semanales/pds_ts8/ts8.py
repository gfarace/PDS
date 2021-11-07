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
#%% 
#Probar que el periodograma es asintoticamente insesgado
#y su varianza no se reduce a cero a pesar de aumentar N

fs = 1000 # frecuencia de muestreo (Hz)

N = np.array([10, 50, 100, 250, 500, 1000, 5000], dtype=float)
R = 200 # realizaciones

sesgo = np.zeros(len(N))
varianza = np.zeros(len(N))
sigma = 2

for i in range(0,len(N)):
    K = int(N[i])
    frec = np.fft.fftfreq(K, d=1/fs)
    x = np.random.normal(0, np.sqrt(sigma), size=(K,R)) # ruido normalmente distribuido
    fft_x = np.fft.fft(x, K, axis = 0)
    Per_x = (1/K)*(np.abs(fft_x)**2)
    E_x = Per_x.mean() # media muestral  
    sesgo[i] = sigma - E_x
    varianza[i] = Per_x.var()

datos = [ 
          [sesgo[0], varianza[0]],
          [sesgo[1], varianza[1]], 
          [sesgo[2], varianza[2]],
          [sesgo[3], varianza[3]],
          [sesgo[4], varianza[4]], 
          [sesgo[5], varianza[5]], 
          [sesgo[6], varianza[6]],
        ]
df = DataFrame(datos, columns=['$s_P$', '$v_P$'], index=N)
HTML(df.to_html())

#%%
plt.close('all')

#Senoidal
#x(k)=a1⋅sen(Ω1⋅k)+n(k)
#Ω1=Ω0+(fr.2π)/N
#Ω0 = pi/2

N = 1000  # cantidad de muestras
fs = 1000 # frecuencia de muestreo (Hz)
R = 200 # realizaciones

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

# Variables aleatorias
fr = np.random.uniform(low=-1/2, high=1/2, size=(N,R)) # distribución uniforme
n_3db = np.random.normal(0, np.sqrt(pot_ruido[0]), size=(N,R)) # ruido con SNR 3dB
n_10db = np.random.normal(0, np.sqrt(pot_ruido[1]), size=(N,R)) # ruido con SNR 10dB

ff = (np.pi/2 + fr*(2*np.pi/N))*(fs/(2*np.pi))
x = a1*np.sin(2*np.pi*ff*tt.reshape(N,1))

######### Para SNR=3dB #########
x1 = x + n_3db
#Periodograma
fft_x1 = np.fft.fft(x1, N, axis = 0)
Per_x1 = (1/N)*(np.abs(fft_x1)**2)
#Welch
fw_x1, Pw_x1 = sig.welch(x1, fs, nperseg=fs, axis=0)
fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(frec,Per_x1)
ax1.set_xlim(245,255)
ax1.set_title('Periodograma para SNR={:1.0f}dB'.format(SNR[0]))
ax1.set(xlabel='Frecuencia', ylabel='PSD')
ax2.plot(fw_x1,Pw_x1)
ax2.set_xlim(245,255)
ax2.set_title('Periodograma de Welch para SNR={:1.0f}dB'.format(SNR[0]))
ax2.set(xlabel='Frecuencia', ylabel='PSD')

######### Para SNR=10dB #########
x2 = x + n_10db
#Periodograma
fft_x2 = np.fft.fft(x2, N, axis = 0)
Per_x2 = (1/N)*(np.abs(fft_x2)**2)
#Welch
fw_x2, Pw_x2 = sig.welch(x2, fs, nperseg=fs, axis=0)
fig2, (ax3, ax4) = plt.subplots(1, 2)
ax3.plot(frec,Per_x2)
ax3.set_xlim(245,255)
ax3.set_title('Periodograma para SNR={:1.0f}dB'.format(SNR[1]))
ax3.set(xlabel='Frecuencia', ylabel='PSD')
ax4.plot(fw_x2,Pw_x2)
ax4.set_xlim(245,255)
ax4.set_title('Periodograma de Welch para SNR={:1.0f}dB'.format(SNR[1]))
ax4.set(xlabel='Frecuencia', ylabel='PSD')

#%%
## CON PADDING ##
plt.close('all')

Npad = N*10

frec_p = np.linspace(0, (Npad-1), Npad)*fs/N

######### Para SNR=3dB #########
#Periodograma
fft_x1_p = np.fft.fft(x1, axis = 0, n=Npad)
Per_x1_p = (1/N)*(np.abs(fft_x1_p)**2)
#Welch
fw_x1_p, Pw_x1_p = sig.welch(x1, fs, nperseg=fs, nfft=Npad, axis=0)
fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(frec_p/(Npad/N),Per_x1_p)
ax1.set_xlim(245,255)
ax1.set_title('Periodograma para SNR={:1.0f}dB con zero padding'.format(SNR[0]))
ax1.set(xlabel='Frecuencia', ylabel='PSD')
ax2.plot(fw_x1_p,Pw_x1_p)
ax2.set_xlim(245,255)
ax2.set_title('Periodograma de Welch para SNR={:1.0f}dB con zero padding'.format(SNR[0]))
ax2.set(xlabel='Frecuencia', ylabel='PSD')

######### Para SNR=10dB #########
#Periodograma
fft_x2_p = np.fft.fft(x2, axis = 0, n=Npad)
Per_x2_p = (1/N)*(np.abs(fft_x2_p)**2)
#Welch
fw_x2_p, Pw_x2_p = sig.welch(x2, fs, nperseg=fs, nfft=Npad, axis=0)
fig2, (ax3, ax4) = plt.subplots(1, 2)
ax3.plot(frec_p/(Npad/N),Per_x2_p)
ax3.set_xlim(245,255)
ax3.set_title('Periodograma para SNR={:1.0f}dB con zero padding'.format(SNR[1]))
ax3.set(xlabel='Frecuencia', ylabel='PSD')
ax4.plot(fw_x2_p,Pw_x2_p)
ax4.set_xlim(245,255)
ax4.set_title('Periodograma de Welch para SNR={:1.0f}dB con zero padding'.format(SNR[1]))
ax4.set(xlabel='Frecuencia', ylabel='PSD')

#%%
# Blackman-Tukey
plt.close('all')

black = np.array(wind.blackman(N)).reshape(N,1)
x1_bm = x1*black
x2_bm = x2*black

#Periodograma
fft_bm_x1 = np.fft.fft(x1_bm, N, axis = 0)
Per_bm_x1 = (1/N)*(np.abs(fft_bm_x1)**2)
fft_bm_x2 = np.fft.fft(x2_bm, N, axis = 0)
Per_bm_x2 = (1/N)*(np.abs(fft_bm_x2)**2)

fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(frec,Per_bm_x1)
ax1.set_xlim(245,255)
ax1.set_title('SNR={:1.0f}dB'.format(SNR[0]))
ax1.set(xlabel='Frecuencia', ylabel='PSD')
ax2.plot(frec,Per_bm_x2)
ax2.set_xlim(245,255)
ax2.set_title('SNR={:1.0f}dB'.format(SNR[1]))
ax2.set(xlabel='Frecuencia', ylabel='PSD')
fig1.suptitle('Periodograma con ventana de Blackman Tukey')

#%%
# Busco el pico de cada realizacion, y la frecuencia correspondiente
plt.close('all')

# Periodograma para SNR=3dB
Per_x1_frec_pos = Per_x1[500:,:] #Me quedo con la parte positiva de frecuencia
frec_pos = frec[:500]
picos_x1 = np.zeros(len(Per_x1_frec_pos[0]))
frec_x1 = np.zeros(len(Per_x1_frec_pos[0]))

for i in range(0,len(Per_x1_frec_pos[0])):
    picos_x1[i]  = np.amax(Per_x1_frec_pos[:,i])
    frec_maximo = np.where(Per_x1_frec_pos[:,i]==picos_x1[i])[0][0]
    frec_x1[i] = frec_pos[frec_maximo]

# Welch para SNR=3dB
picos_w_x1 = np.zeros(len(Pw_x1[0]))
frec_w_x1 = np.zeros(len(Pw_x1[0]))

for i in range(0,len(Pw_x1[0])):
    picos_w_x1[i]  = np.amax(Pw_x1[:,i])
    frec_maximo = np.where(Pw_x1[:,i]==picos_w_x1[i])[0][0]
    frec_w_x1[i] = fw_x1[frec_maximo]

# Blackman Tukey para SNR=3dB
Per_bm_x1_mitad = Per_bm_x1[0:500,:] #Me quedo con la mitad
picos_bm_x1 = np.zeros(len(Per_bm_x1[0]))
frec_bm_x1 = np.zeros(len(Per_bm_x1[0]))

for i in range(0,len(Pw_x1[0])):
    picos_bm_x1[i]  = np.amax(Per_bm_x1_mitad[:,i])
    frec_maximo = np.where(Per_bm_x1_mitad[:,i]==picos_bm_x1[i])[0][0]
    frec_bm_x1[i] = frec_pos[frec_maximo]


# Periodograma para SNR=10dB
Per_x2_frec_pos = Per_x2[500:,:] #Me quedo con la parte positiva de frecuencia
picos_x2 = np.zeros(len(Per_x2_frec_pos[0]))
frec_x2 = np.zeros(len(Per_x2_frec_pos[0]))

for i in range(0,len(Per_x2_frec_pos[0])):
    picos_x2[i]  = np.amax(Per_x2_frec_pos[:,i])
    frec_maximo = np.where(Per_x2_frec_pos[:,i]==picos_x2[i])[0][0]
    frec_x2[i] = frec_pos[frec_maximo]

# Welch para SNR=10dB
picos_w_x2 = np.zeros(len(Pw_x2[0]))
frec_w_x2 = np.zeros(len(Pw_x2[0]))

for i in range(0,len(Pw_x2[0])):
    picos_w_x2[i]  = np.amax(Pw_x2[:,i])
    frec_maximo = np.where(Pw_x2[:,i]==picos_w_x2[i])[0][0]
    frec_w_x2[i] = fw_x2[frec_maximo]

# Blackman Tukey para SNR=10dB
Per_bm_x2_mitad = Per_bm_x2[0:500,:] #Me quedo con la mitad
picos_bm_x2 = np.zeros(len(Per_bm_x2[0]))
frec_bm_x2 = np.zeros(len(Per_bm_x2[0]))

for i in range(0,len(Pw_x1[0])):
    picos_bm_x2[i]  = np.amax(Per_bm_x2_mitad[:,i])
    frec_maximo = np.where(Per_bm_x2_mitad[:,i]==picos_bm_x2[i])[0][0]
    frec_bm_x2[i] = frec_pos[frec_maximo]


indice = ['$\hat{\Omega_{1}}^{X}$', 'frec $\hat{\Omega_{1}}^{X}$', '$\hat{\Omega_{1}}^{W}$', 'frec $\hat{\Omega_{1}}^{W}$','$\hat{\Omega_{1}}^{BT}$', 'frec $\hat{\Omega_{1}}^{BT}$']

datos = [ 
            [np.mean(picos_x1), np.mean(picos_x2)],
            [np.mean(frec_x1), np.mean(frec_x2)],
            [np.mean(picos_w_x1), np.mean(picos_w_x2)],
            [np.mean(frec_w_x1), np.mean(frec_w_x2)],
            [np.mean(picos_bm_x1), np.mean(picos_bm_x2)],
            [np.mean(frec_bm_x1), np.mean(frec_bm_x2)]
        ]
df = DataFrame(datos, columns=['3dB','10dB'], index=indice)
HTML(df.to_html())

