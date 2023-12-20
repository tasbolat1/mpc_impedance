import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pickle.load(open('some_file_to_read.pkl', 'rb'))


fig, ax = plt.subplots(nrows=4, figsize=(16,9))
ax[0].plot(data['_time'], data['_x'][:,3], label='theta_dot')
ax[0].legend()
ax[1].plot(data['_time'], data['_aux'][:,1:3], label='pose_des', linestyle='--')
ax[1].legend()
ax[2].plot(data['_time'], data['_x'][:,2], label='theta_des')
ax[2].legend()
ax[3].plot(data['_time'], data['_x'][:,0:2], label='pose')
ax[3].legend()
plt.show()

fig, ax = plt.subplots(nrows=1, figsize=(16,9))
ax.plot(data['_x'][:,0], data['_x'][:,1], label='theta_dot')
plt.show()