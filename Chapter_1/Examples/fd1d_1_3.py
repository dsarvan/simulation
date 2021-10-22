#!/usr/bin/env python
# File: fd3d_1_3.py
# Name: D.Saravanan
# Date: 21/10/2021

""" Simulation of a pulse hitting a dielectric medium """

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 8,
            'axes.labelsize': 10, 'axes.titlesize': 10, 'figure.titlesize': 10})

ke = 200
ex = np.zeros(ke)
hy = np.zeros(ke)
