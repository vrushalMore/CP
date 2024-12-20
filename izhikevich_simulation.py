import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

T = 1000

Ne = 800

Ni = 200

re = np.random.rand(Ne,1)
ri = np.random.rand(Ni,1)

p_RS = [0.02, 0.2, -65, 8, "regular spiking (RS)"]
p_IB = [0.02, 0.2, -55, 4, "intrinsically bursting (IB)"]
p_CH = [0.02, 0.2, -51, 2, "chattering (CH)"]
p_FS = [0.1, 0.2, -65, 2, "fast spiking (FS)"]
p_TC = [0.1, 0.25, -65, 0.05, "thalamic-cortical (TC)"]
p_LTS = [0.02, 0.25, -65, 2, "low-threshold spiking (LTS)"] 
p_RZ  = [0.1, 0.26, -65, 2, "resonator (RZ)"]

a_e, b_e, c_e, d_e, name_e = p_RS
a_i, b_i, c_i, d_i, name_i = p_LTS

a = np.vstack((a_e * np.ones((Ne, 1)), a_i + 0.08 * ri))
b = np.vstack((b_e * np.ones((Ne, 1)), b_i - 0.05 * ri))
c = np.vstack((c_e + 15 * re**2,       c_i * np.ones((Ni, 1))))
d = np.vstack((d_e-6 * re**2,          d_i * np.ones((Ni, 1))))
S = np.hstack((0.5 * np.random.rand(Ne+Ni, Ne), -1*np.random.rand(Ne+Ni, Ni)))

v = -65 * np.ones((Ne+Ni, 1))
u = b * v
firings = np.array([]).reshape(0, 2)

I_array = np.zeros((Ne+Ni, T))
v_array = np.zeros((Ne+Ni, T))
u_array = np.zeros((Ne+Ni, T))

for t in range(0, T):
    I = np.vstack((5 * np.random.randn(Ne, 1), 2 * np.random.randn(Ni, 1)))
    if t > 0:  
        I += np.sum(S[:, fired], axis=1).reshape(-1, 1)
    v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
    v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
    u += a * (b * v - u)
    fired = np.where(v >= 30)[0]
    if fired.size > 0:
        firings = np.vstack((firings, np.hstack((t * np.ones((fired.size, 1)), fired.reshape(-1, 1)))))
    v[fired] = c[fired]
    u[fired] = u[fired] + d[fired]
    I_array[:, t] = I.flatten()
    v_array[:, t] = v.flatten()
    u_array[:, t] = u.flatten()

plt.figure(figsize=(7, 7))
plt.scatter(firings[:, 0], firings[:, 1], s=1, c='k')
excitatory = firings[:, 1] < Ne
inhibitory = firings[:, 1] >= Ne
plt.axhline(y=Ne, color='k', linestyle='-', linewidth=1)
plt.text(0.8, 0.76, 'excitatory', color='k', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1)) 
plt.text(0.8, 0.84, 'inhibitory', color='k', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1))
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.xlim([0, T])
plt.ylim([0, Ne+Ni])
plt.yticks(np.arange(0, Ne+Ni+1, 200))
plt.tight_layout()
plt.show()
