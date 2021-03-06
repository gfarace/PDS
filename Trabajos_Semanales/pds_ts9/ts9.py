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
array_latidos_n = array_latidos/np.amax(array_latidos)

plt.figure(1)
plt.plot(tiempo, array_latidos_n)
plt.title('Latidos presentes en el registro')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud normalizada')

# Busco los valores en las muestras 200 (maximos) y 350 (minimos)
# Quiero agrupar y separar en dos grupos
maximos = array_latidos[200,:]
minimos = array_latidos[350,:]

cota_x = 15000
cota_y = -2000

plt.figure(2)
plt.scatter(maximos,minimos)
plt.axvline(x=cota_x, color='r')
plt.axhline(y=cota_y, color='r')
plt.title('Agrupación por tipo de latido')

nn = np.bitwise_and(maximos < cota_x, minimos > cota_y)
vv = ~nn

lat_vent = array_latidos[:,vv]
lat_norm = array_latidos[:,nn]
norm_prom = np.mean(lat_norm , axis=1)
vent_prom = np.mean(lat_vent , axis=1)

plt.figure(3)
vent = plt.plot(lat_vent/np.amax(array_latidos), 'g')
norm = plt.plot(lat_norm/np.amax(array_latidos), 'b')
vent_p = plt.plot(vent_prom/np.amax(array_latidos), '--y',lw=2)
norm_p = plt.plot(norm_prom/np.amax(array_latidos), '--r',lw=2)
plt.title('Tipos de latidos')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud normalizada')
plt.legend(['Ventricular','Normal'])

#%%
plt.close('all')

# Me quedo con un rango de latidos
inicio = 0
fin = 50

lat_vent_c = lat_vent[:,inicio:fin]
lat_norm_c = lat_norm[:,inicio:fin]

lv_prom = np.mean(lat_vent_c, axis=1)
ln_prom = np.mean(lat_norm_c, axis=1)

plt.figure(1)
plt.plot(lat_vent_c/np.amax(array_latidos), 'g')
plt.plot(lat_norm_c/np.amax(array_latidos), 'b')
plt.plot(lv_prom/np.amax(array_latidos), '--y',lw=2)
plt.plot(ln_prom/np.amax(array_latidos), '--r',lw=2)
plt.title('{:1.0f} latidos del registro'.format(fin-inicio))
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud normalizada')
plt.legend(['Ventricular','Normal'])

#%%

# La mediana representa mejor el latido normal, el promedio no representa
# ningun valor. La mediana, sin embargo, elimina los latidos ventriculares

vent_pad = np.pad(lat_vent_c, pad_width=((2000,2000),(0,0)), mode='constant')
norm_pad = np.pad(lat_norm_c, pad_width=((2000,2000),(0,0)), mode='constant')

vent_pad_prom = np.mean(vent_pad,axis=1)
norm_pad_prom = np.mean(norm_pad,axis=1)

plt.figure(1)
plt.plot(vent_pad/np.amax(array_latidos), 'g')
plt.plot(norm_pad/np.amax(array_latidos), 'b')
plt.plot(vent_pad_prom/np.amax(array_latidos), '--y',lw=2)
plt.plot(norm_pad_prom/np.amax(array_latidos), '--r',lw=2)
plt.title('{:1.0f} latidos con padding'.format(fin-inicio))
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud normalizada')
plt.legend(['Ventricular','Normal'])

#%%
plt.close('all')
# Calculo la densidad espectral de potencia para cada realizacion 
# y promedio los espectros
# Aplico Welch
N = len(vent_pad)
fw_v, Pw_v = sig.welch(vent_pad, fs, nperseg = N/2, axis=0)
fw_n, Pw_n = sig.welch(norm_pad, fs, nperseg = N/2, axis=0)

norm = np.amax(Pw_v)

Pw_v_prom = np.mean(Pw_v,axis=1)
Pw_n_prom = np.mean(Pw_n,axis=1)

plt.figure(1)
plt.plot(fw_v,Pw_v/norm, 'g')
plt.plot(fw_n,Pw_n/norm, 'b')
plt.plot(fw_v,Pw_v_prom/norm, '--y',lw=2)
plt.plot(fw_n,Pw_n_prom/norm, '--r',lw=2)
plt.title('Espectro aplicando Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [$V^{2}/Hz$]')
plt.xlim(0,20)
plt.legend(['Ventricular','Normal'])

#%%
plt.close('all')
# Uso el promedio de los espectros
# Calculo la potencia
corte_energia = 0.99 #La señal esta muy limpia, no tiene mucho ruido 

Pot_n = np.cumsum(Pw_n_prom)/np.sum(Pw_n_prom)
corte_n = np.where(Pot_n >corte_energia)[0][0]

Pot_v = np.cumsum(Pw_v_prom)/np.sum(Pw_v_prom)
corte_v = np.where(Pot_v >corte_energia)[0][0]

plt.figure(1)
plt.plot(fw_n,Pw_n_prom/norm, 'k')
plt.fill_between(fw_n, 0, Pw_n_prom/norm, where = fw_v < fw_v[corte_n], color='blue')
plt.title('Ancho de banda donde se concentra el {:3.0f}% de la energia para pulsos normales'.format(corte_energia*100))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [$V^{2}/Hz$]')
plt.xlim(0,50)

plt.annotate(   "BW_n = {:3.1f} Hz".format(fw_n[corte_n]),
                xy=(fw_n[corte_n], Pw_n_prom[corte_n]/norm),
                xytext=(-20,20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle='->')
)

plt.figure(2)
plt.plot(fw_v,Pw_v_prom/norm, 'k')
plt.fill_between(fw_n, 0, Pw_v_prom/norm, where = fw_v < fw_v[corte_v], color='green')
plt.title('Ancho de banda donde se concentra el {:3.0f}% de la energia para pulsos ventriculares'.format(corte_energia*100))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [$V^{2}/Hz$]')
plt.xlim(0,50)

plt.annotate(   "BW_n = {:3.1f} Hz".format(fw_v[corte_v]),
                xy=(fw_v[corte_v], Pw_v_prom[corte_v]/norm),
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


# # Defina la plantilla del filtro

# fs0 = ?? # fin de la banda de detenida 0
# fc0 = ?? # comienzo de la banda de paso
# fc1 = ?? # fin de la banda de paso
# fs1 = ?? # comienzo de la banda de detenida 1



