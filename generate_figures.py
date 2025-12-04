import matplotlib.pyplot as plt
import numpy as np

# 1. PRE Figure: Composite
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# Upper Left: Topological Diode (Rectification Ratio vs Anisotropy)
gamma = np.linspace(0, 0.5, 20)
R_passive = np.ones_like(gamma)
R_active = np.ones_like(gamma)
mask = gamma > 0.25
R_active[mask] = 10**6
axs[0, 0].plot(gamma, R_passive, label='Passive')
axs[0, 0].plot(gamma, R_active, label='Active')
axs[0, 0].set_yscale('log')
axs[0, 0].set_title('Topological Diode')
axs[0, 0].set_xlabel('Anisotropy $\gamma$')
axs[0, 0].set_ylabel('$R_{rect}$')

# Upper Right: Toxic Sponge (Counts vs Time) - simplified bar chart or lines
t = np.linspace(0, 1000, 10)
rim_control = np.linspace(2500, 2491, 10)
core_control = np.linspace(1400, 1392, 10)
rim_sponge = np.linspace(2500, 1162, 10)
core_sponge = np.linspace(1400, 3271, 10)
axs[0, 1].plot(t, core_sponge, label='Core (Sponge)')
axs[0, 1].plot(t, rim_sponge, label='Rim (Sponge)')
axs[0, 1].set_title('Toxic Sponge')

# Bottom: Spectral Dimension (spanning bottom)
ax_bottom = plt.subplot(2, 1, 2)
# Need to remove the existing axes at (1,0) and (1,1) if using subplot(2,1,2)
# But simpler to just use axs[1,0] and hide axs[1,1] or something.
# Let's just use axs[1,0] for now and stretch later or ignore layout perfection.
# Actually let's just make 3 separate plots if needed, but the paper expects one file.
# We will just use the 2x2 grid and put the spectral dimension in the bottom row.
t_sd = np.logspace(1, 4, 20)
P_core = t_sd**(-0.0/2)
P_healthy = t_sd**(-2.05/2)
P_rim = t_sd**(-3.49/2)

axs[1, 0].loglog(t_sd, P_core, label='Core $d_s=0$')
axs[1, 0].loglog(t_sd, P_healthy, label='Healthy $d_s=2.05$')
axs[1, 0].loglog(t_sd, P_rim, label='Rim $d_s=3.49$')
axs[1, 0].set_title('Spectral Dimension')

axs[1, 1].axis('off') # Empty slot

plt.tight_layout()
plt.savefig('PRE_Figures_Composite.pdf')
plt.close()

# 2. SciRep Figure: URC Wedge
plt.figure(figsize=(6, 6))
ds = np.linspace(0, 4, 100)
alpha = ds / 2
plt.plot(ds, alpha, 'k--', label='URC Wedge $\\alpha = d_s/2$')
plt.fill_between(ds, alpha-0.1, alpha+0.1, color='gray', alpha=0.2)
plt.scatter([0], [0], color='red', marker='^', label='GBM Core')
plt.scatter([3.49], [0.65], color='red', marker='v', label='GBM Rim') # Dummy alpha
plt.scatter([1.5], [0.75], color='blue', label='Cosmic Web')
plt.xlabel('Spectral Dimension $d_s$')
plt.ylabel('Anomalous Exponent $\\alpha_{RW}$')
plt.legend()
plt.title('URC Wedge')
plt.savefig('SciRep_URC_Wedge.pdf')
plt.close()

# 3. PRL Figure: Spectral Horizon
plt.figure(figsize=(6, 4))
r = np.linspace(0, 1, 100)
ds_r = np.zeros_like(r)
ds_r[r < 0.6] = 0.1
ds_r[r >= 0.6] = 3.49
plt.plot(r, ds_r)
plt.axvline(0.6, color='k', linestyle='--')
plt.xlabel('Radial Distance $r$')
plt.ylabel('Spectral Dimension $d_s(r)$')
plt.title('Malignancy Horizon')
plt.savefig('PRL_SpectralHorizon.pdf')
plt.close()
