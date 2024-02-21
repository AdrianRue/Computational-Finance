import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

S0 = 100
K = 99
r = .06
sigma = 0.2
T = 1
M =1000
dt = T/M

 
S = [S0]
Z_m = np.random.normal(size = M)
for m in Z_m:
    S.append(S[-1]+r*S[-1]*dt+sigma*S[-1]*np.sqrt(dt)*m)

plt.plot(np.arange(0, T+dt, dt), S)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()

    