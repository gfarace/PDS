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
import scipy.io as sio

# Para la tabla
from pandas import DataFrame
from IPython.display import HTML
#%%

mat_struct = sio.loadmat('ECG_TP4.mat')

plt.close('all')

fs = 1000

ecg = mat_struct['ecg_lead']
qrs_detections = mat_struct['qrs_detections']

x = 250
y = 300

ecg_matr = np.hstack[ ecg[int(ii-x):int(ii+y)].reshape(win_s, 1) for ii in qrs_detections ]