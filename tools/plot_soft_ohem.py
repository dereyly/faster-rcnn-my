import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0,1,0.01)
y=np.minimum(1-x,np.sqrt(x)+0.05)

plt.plot(x,y)
plt.ylabel('some numbers')
plt.show()
