
#plot_parabola.py


import numpy as np
import matplotlib.pyplot as plt 

x = np.linspace(-100,100,10000)
y = x*x

#Plotting
plt.plot(x,y,linewidth = 5)
plt.xlabel('x',fontweight='bold',fontsize=12)
plt.ylabel('y = x^2',fontweight='bold',fontsize=12)
plt.title('Wave',fontweight='bold',fontsize=12)
plt.grid(True)
plt.show()