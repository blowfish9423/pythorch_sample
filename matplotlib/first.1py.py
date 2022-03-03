import matplotlib.pyplot as plt 
import numpy as np
#x = list(range(0,10))
#y = list(range(-10,0))
x = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
#y = np.array([-9,-7,-5,-3,-1,1,3,5,7,9,11])

plt.xlabel("X value")
plt.ylabel("Function value")
plt.title(" Function demo")

y = 2*x+1
plt.plot(x,y,label="y = 2*x+1")

y = 3*x+1
plt.plot(x,y,label="y = 3*x+1")

plt.grid()
plt.legend(loc="best",shadow=True,fontsize="small")

plt.show()

