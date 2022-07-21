#!/usr/bin/env python
# File: fdtdgui.py
# Name: D.Saravanan
# Date: 21/07/2022

""" Script to create a graphical user interface for the FDTD program """

import tkinter as tk

# create Tk object for root window of the application
root = tk.Tk()

root.title("FTDT")
root.geometry("800x600")
root.resizable(False, False)

label = tk.Label(root, text="Computational Electrodynamics")
input = tk.Entry(root)

# start event loop (infinite loop)
root.mainloop()
