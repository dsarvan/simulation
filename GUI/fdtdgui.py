#!/usr/bin/env python
# File: fdtdgui.py
# Name: D.Saravanan
# Date: 21/07/2022

""" Script to create a graphical user interface for the FDTD program """

import sys
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk



# create Tk object for root window of the application
root = tk.Tk()

root.title("FTDT")
root.geometry("800x600")
root.resizable(False, False)
root.rowconfigure(index=2, weight=1)
root.columnconfigure(index=1, weight=1)


# Insert a menu bar on the root window
menubar = tk.Menu(root)
root.config(menu=menubar)

# Create a menu button labeled "File" that brings up a menu
filemenu = tk.Menu(menubar)
menubar.add_cascade(label="File", menu=filemenu)

# Create entries in the "File" menu
def doPrint(): print('doPrint')
def doSave(): print('doSave')
filemenu.add_command(label="Print", command=doPrint)
filemenu.add_command(label="Save", command=doSave)
filemenu.add_separator()
filemenu.add_command(label='Quit', command=sys.exit)


# Create a menu button labeled "Edit" that brings up a menu
editmenu = tk.Menu(menubar)
menubar.add_cascade(label="Edit", menu=filemenu)

# Create entries in the "Edit" menu
def doPrint(): print('doPrint')
def doSave(): print('doSave')
filemenu.add_command(label="Print", command=doPrint)
filemenu.add_command(label="Save", command=doSave)
filemenu.add_separator()
filemenu.add_command(label='Quit', command=sys.exit)






# Processing unit
processing = ["CPU process", "GPU process"]
process_label = tk.Label(root, text="Processing: ")
process_input = tk.Listbox(root, height=1)

for process in processing:
    process_input.insert(tk.END, process)
    
process_label.grid(row=0, column=0, padx=5, pady=5)
process_input.grid(row=0, column=1, sticky=tk.W)

#process_label.grid(row=0, column=0, sticky=tk.E + tk.W, padx=5, pady=5)
#process_input.grid(row=0, column=1, sticky=tk.E + tk.W)

# Dimension selection
dimensions = ["One-Dimensional Simulation", "Two-Dimensional Simulation", "Three-Dimensioanl Simulation"] 
dimensions_label = tk.Label(root, text="Dimensions: ")
dimensions_input = tk.Listbox(root, height=1)

for dimension in dimensions:
    dimensions_input.insert(tk.END, dimension)

#dimensions_label.grid(row=1, column=0, sticky=tk.E + tk.W, padx=5, pady=5)
#dimensions_label.grid(row=1, column=0, sticky=tk.E + tk.W, padx=5, pady=5)

dimensions_label.grid(row=1, column=0, padx=5, pady=5)
dimensions_input.grid(row=1, column=1, sticky=tk.W)

# Pulse parameters
time_label = tk.Label(root, text="Time: ")
time_input = tk.Entry(root)

time_label.grid(row=2, column=0, padx=5, pady=5)
time_input.grid(row=2, column=1, sticky=tk.W)


fig = Figure(figsize = (5, 5), dpi = 100)

canvas = FigureCanvasTkAgg(fig, root)
canvas.draw()
canvas.get_tk_widget().grid(row=3, column=1, sticky=tk.W)










message_input = tk.Text(root)
message_input.grid(row=6, column=0, columnspan=2, sticky="news", pady=5)

start_button = tk.Button(root, text="Start")
start_button.grid(row=10, column=1, sticky=tk.E, padx=5, pady=5)


def applicationInfo():

    process = process_input.get()
    dimension = dimensions_input.curselection()
    message = message_input.get("1.0", tk.END)

    filename = f"{process}_{dimension}.txt"

    with open(filename, "w") as fh:
        fh.write(message)


# start event loop (infinite loop)
root.mainloop()
