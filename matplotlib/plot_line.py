
#plot_line.py


import numpy as np
import matplotlib.pyplot as plt 

x = np.linspace(0, 2, 100)
y0 = 2*x
y1 = 2*x+7
#y1 = x+5
#y2 = x**2
#y3 = x**3
# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(x, y0, label='linear')  # Plot some data on the axes.
ax.plot(x, y1, label='linear')  # Plot some data on the axes.
#ax.plot(x, y2, label='quadratic')  # Plot more data on the axes...
#ax.plot(x, y3, label='cubic')  # ... and some more.
ax.set_xlabel('x label')  # Add an x-label to the axes.
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.show()