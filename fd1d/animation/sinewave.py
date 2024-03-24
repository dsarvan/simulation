#!/usr/bin/env python
# File: sinewave.py
# Name: D.Saravanan
# Date: 10/01/2022

""" Script to create an animation of a sine wave """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 1000
x = np.linspace(0, 6*np.pi, n)
y = np.sin(x)

# define the meta data for the movie
fwriter = animation.writers["ffmpeg"]
data = dict(title = "Sine wave", artist = "Saran", comment = "Animation of a sine wave")
writer = fwriter(fps = 15, metadata = data) 

# plot the sine wave line
fig, ax = plt.subplots()
sine_line, = ax.plot(x, y, 'b')
red_circle, = ax.plot([], [], 'ro', markersize=10)
ax.set(xlim=(0, 6*np.pi), xlabel=r'x', ylabel=r'sin(x)')

# update the frames for the movie
with writer.saving(fig, 'sinewave.mp4', 100):
    for i in range(n):
        x0 = x[i]; y0 = y[i]
        red_circle.set_data(x0, y0)
        writer.grab_frame()
