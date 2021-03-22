#plotsin.py

import numpy as np
import matplotlib.pyplot as plt 

x = np.linspace(0, 2 * np.pi,100)
y = np.sin(x)

#Plotting
plt.plot(x,y,linewidth = 3)
plt.xlabel('x',fontweight='bold',fontsize=12)
plt.ylabel(r'$\sin x$',fontweight='bold',fontsize=12)
plt.title('Wave',fontweight='bold',fontsize=12)
plt.grid(True)
plt.show()