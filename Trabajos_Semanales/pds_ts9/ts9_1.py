#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/11/2021

@author: Gisela Farace

Descripción: Tarea semanal 9
------------
"""

# Importación de módulos para Jupyter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.signal.windows as wind
import scipy.io as sio

# Para la tabla
from pandas import DataFrame
from IPython.display import HTML
#

#%%
# ecg_lead: Registro de ECG muestreado a $fs=1$ KHz durante una prueba de esfuerzo
# qrs_pattern1: Complejo de ondas QRS normal
# heartbeat_pattern1: Latido normal
# heartbeat_pattern2: Latido de origen ventricular
# qrs_detections: vector con las localizaciones (en # de muestras) donde ocurren los latidos

plt.close('all')
mat_struct = sio.loadmat('/media/sf_UTN/PDS/PDS/Trabajos_Semanales/pds_ts9/ECG_TP4.mat')

fs = 1000

ecg = mat_struct['ecg_lead']
qrs = mat_struct['qrs_detections']
patron_normal = mat_struct['heartbeat_pattern1']
patron_ventricular = mat_struct['heartbeat_pattern2']

x = 200 #antes del pico
y = 350 #despues del pico

ecg_matr = [ (ecg[int(ii-x):int(ii+y)]) for ii in qrs]
tiempo = np.arange(0,x+y,1)

array_latidos = np.hstack(ecg_matr)
# Alineacion de latidos
array_latidos = array_latidos - np.mean(array_latidos, axis=0)
# Normalizo por el maximo
array_latidos_n = array_latidos/np.amax(array_latidos)
prom_latidos = np.mean(array_latidos_n , axis=1)
media_latidos = np.median(array_latidos_n , axis=1)

plt.figure(1)
plt.plot(tiempo, array_latidos_n)
plt.plot(tiempo,prom_latidos, '--k',label='Promedio',lw=4)
plt.plot(tiempo,media_latidos, '--b',label='Mediana',lw=4)
plt.title('Latidos presentes en el registro')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud normalizada')
plt.legend()

# Me quedo con un rango de latidos
inicio = 0
fin = 50
array_latidos_c = array_latidos_n[:,inicio:fin]
prom_latidos_c = np.mean(array_latidos_c , axis=1)
media_latidos_c = np.median(array_latidos_c , axis=1)

plt.figure(2)
plt.plot(tiempo, array_latidos_c)
plt.plot(tiempo,prom_latidos_c, '--k',label='Promedio',lw=4)
plt.plot(tiempo,media_latidos_c, '--b',label='Mediana',lw=4)
plt.title('{:1.0f} latidos del registro'.format(fin-inicio))
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud normalizada')
plt.legend()

# La mediana representa mejor el latido normal, el promedio no representa
# ningun valor. La mediana, sin embargo, elimina los latidos ventriculares

# Hago padding del rango de latidos usando la mediana
latidos_pad = np.pad(array_latidos_c , pad_width=((2000,2000),(0,0)), mode='constant')
latidos_pad_med = np.median(latidos_pad,axis=1)
latidos_pad_prom = np.mean(latidos_pad,axis=1)

plt.figure(3)
plt.plot(latidos_pad)
plt.plot(latidos_pad_med,'--k',label='Promedio',lw=4)
plt.plot(latidos_pad_prom,'--b',label='Mediana',lw=4)
plt.title('{:1.0f} latidos con padding'.format(fin-inicio))
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud normalizada')
plt.legend()

# Calculo la densidad espectral de potencia para cada realizacion 
# y promediar los espectros
# Aplico Welch
N = len(latidos_pad)
fw, Pw = sig.welch(latidos_pad, fs, nperseg = N/2, axis=0)
fw_m, Pw_m = sig.welch(latidos_pad_med, fs, nperseg = N/2, axis=0)
fw_p, Pw_p = sig.welch(latidos_pad_prom, fs, nperseg = N/2, axis=0)

#Hago la mediana del espectro de Welch
mediana_Pw = np.median(Pw,axis=1)

plt.figure(4)
plt.plot(fw,Pw)
plt.plot(fw,mediana_Pw,'--r',label='Mediana del espectro',lw=4)
plt.plot(fw_p,Pw_m,'--k',label='Espectro del promedio',lw=4)
plt.plot(fw_p,Pw_p,'--b',label='Espectro de la mediana',lw=4)
plt.title('Espectro aplicando Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [$V^{2}/Hz$]')
plt.xlim(0,20)
plt.legend()


#%%
# Uso la mediana de los espectros
# Calculo la potencia
Potencia = np.cumsum(mediana_Pw)/np.sum(mediana_Pw)
corte_energia = 0.99 #La señal esta muy limpia, no tiene mucho ruido 
corte = np.where(Potencia>corte_energia)[0][0]

plt.figure(5)
plt.plot(fw,mediana_Pw, 'k')
plt.fill_between(fw, 0, mediana_Pw, where = fw < fw[corte], color='orange')
plt.xlim(0,50)

plt.annotate(   "BW = {:3.1f} Hz".format(fw[corte]),
                xy=(fw[corte], mediana_Pw[corte]),
                xytext=(-20,20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle='->')
)

# Anular a partir de la frecuencia[corte], hacer la antitransformada y 
# ver como queda el ECG (no tendria la fase)
# Filtrado de fase cero:
# Latido mediano --> FFT (modulo y fase) --> Anular en frec[corte] --> reconstruir 

#%%
#Busco maximos y minimos de cada realizacion para poder separar
plt.close('all')

maximos = array_latidos[200,:]
minimos = array_latidos[350,:]

plt.figure(1)
plt.scatter(maximos,minimos)
# [15000, -2000]

nn = np.bitwise_and(maximos < 15000, minimos > -2000)
vv = ~nn

plt.figure(2)
plt.plot(array_latidos[:,vv], 'g')
plt.plot(array_latidos[:,nn], 'b')


## 
# patron_normal = mat_struct['heartbeat_pattern1']
# patron_ventricular = mat_struct['heartbeat_pattern2']

# plt.plot(patron_normal)
# plt.plot(patron_ventricular)



#%%
plt.close('all')
#latidos = np.hstack(ecg_matr)
array_latidos = np.hstack(ecg_matr)
array_latidos = array_latidos - np.mean(array_latidos, axis=0)

array_latidos_pad = np.pad(array_latidos, pad_width=((2000,2000),(0,0)), mode='constant')
N = len(array_latidos_pad)
fw, Pw = sig.welch(array_latidos_pad, fs, nperseg = N/2, axis=0)
# plt.plot(fw,Pw)

area = np.cumsum(Pw, axis=0)/np.sum(Pw, axis=0)

aa = area[12,:]
bb = area[35,:]

plt.figure(1)
plt.scatter(aa,bb)
# [0,6 : 0,85]

nn = np.bitwise_and(aa < 0.6, bb < 0.85)
vv = ~np.bitwise_and(aa < 0.6, bb < 0.85)

plt.figure(2)
plt.plot(array_latidos[:,vv], 'g')
plt.plot(array_latidos[:,nn], 'b')





# # Defina la plantilla del filtro

# fs0 = ?? # fin de la banda de detenida 0
# fc0 = ?? # comienzo de la banda de paso
# fc1 = ?? # fin de la banda de paso
# fs1 = ?? # comienzo de la banda de detenida 1



