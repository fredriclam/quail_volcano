import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('log_data.dat', 'rb') as f:
    theta_store = pickle.load(f)

thetas = [np.squeeze(np.array(arr)) for arr in theta_store]

i = 2

plt.clf()
C = plt.contourf(thetas[i],levels=100) 
plt.colorbar(C)
plt.show()

# or


plt.clf()
p = plt.plot(thetas[i].T)
plt.show()

# Later plot
half_index = int(np.shape(thetas[i])[0]/2)
plt.clf()
p = plt.plot(thetas[i].T[:,half_index:])
plt.show()