import numpy as np
import matplotlib.pyplot as plt
import gaussian as g

std = np.geomspace(0.01, 7, 100, endpoint=True)
std_kern, std_pulse = np.meshgrid(std, std, indexing='xy')
results = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        kernel = g.Gaussian(std_kern[i,j], 25)
        kernel -= np.mean(kernel)
        kernel /= (np.sum(kernel**2))
        pulse = g.Gaussian(std_pulse[i,j], 25)
        pulse -= np.mean(pulse)
        results[i,j] = np.sum(kernel * pulse)

plt.pcolor(std_kern, std_pulse, results)
plt.xlim([std[0], std[-1]])
plt.ylim([std[0], std[-1]])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('kernel sigma')
plt.ylabel('Pulse sigma')
plt.show()