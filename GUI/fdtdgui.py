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
#root.configure(bg="white")
root.resizable(False, False)
root.rowconfigure(index=2, weight=1)
root.columnconfigure(index=1, weight=1)

process_label = tk.Label(root, text="Processing: ")
process_input = tk.Entry(root)

process_label.grid(row=0, column=0, sticky=tk.E + tk.W, padx=5, pady=5)
process_input.grid(row=0, column=1, sticky=tk.E + tk.W)


dimensions = ["One-Dimensional Simulation", "Two-Dimensional Simulation", "Three-Dimensioanl Simulation"] 
dimensions_label = tk.Label(root, text="Dimensions: ")
dimensions_input = tk.Listbox(root, height=1)

for dimension in dimensions:
    dimensions_input.insert(tk.END, dimension)

dimensions_label.grid(row=1, column=0, sticky=tk.E + tk.W, padx=5, pady=5)
dimensions_input.grid(row=1, column=1, sticky=tk.E + tk.W)


message_input = tk.Text(root)
message_input.grid(row=2, column=0, columnspan=2, sticky="news", pady=5)

start_button = tk.Button(root, text="Start")
start_button.grid(row=10, column=1, sticky=tk.E, padx=5, pady=5)

# start event loop (infinite loop)
root.mainloop()
