import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "fk48_canonical_raw.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

h = np.array(DATA["h"], dtype=float)
gg_over_h4 = np.array(DATA["Kgg_singular_values_over_h4"], dtype=float)
gg = gg_over_h4 * h**4
gp = np.array(DATA["Kgp_norm_over_h2"], dtype=float) * h**2
restricted = np.array(DATA["restricted_stationary_response_norm"], dtype=float)
full = np.array(DATA["full_stationary_response_norm"], dtype=float)

def fit_power(x, y):
    p, logc = np.polyfit(np.log(x), np.log(y), 1)
    yhat = np.exp(logc) * x**p
    ss_res = np.sum((np.log(y) - np.log(yhat))**2)
    ss_tot = np.sum((np.log(y) - np.mean(np.log(y)))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0
    return p, np.exp(logc), r2

# Figure 1: canonical block hierarchy
pgg, _, rgg = fit_power(h, gg[0])
pgp, _, rgp = fit_power(h, gp)
fig = plt.figure(figsize=(6.4, 4.6))
ax = fig.add_subplot(111)
ax.loglog(h, gg[0], label=fr"$\|K_{{gg}}\|$ ($p={pgg:.4f}$)")
ax.loglog(h, gp, label=fr"$\|K_{{gp}}\|$ ($p={pgp:.4f}$)")
ax.set_xlabel(r"Refinement scale $h$")
ax.set_ylabel("Block norm")
ax.set_title("Canonical FK48 block hierarchy")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figure1_canonical_block_hierarchy.pdf")
fig.savefig(OUT / "figure1_canonical_block_hierarchy.png", dpi=300)
plt.close(fig)

# Figure 2: all four Kgg singular values
fig = plt.figure(figsize=(6.4, 4.6))
ax = fig.add_subplot(111)
for i in range(4):
    p, _, _ = fit_power(h, gg[i])
    ax.loglog(h, gg[i], label=fr"$\sigma_{i+1}$ ($p={p:.4f}$)")
ax.set_xlabel(r"Refinement scale $h$")
ax.set_ylabel(r"Singular value of $K_{gg}$")
ax.set_title(r"Four gauge-sector singular values: $K_{gg}$")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figure2_Kgg_singular_values.pdf")
fig.savefig(OUT / "figure2_Kgg_singular_values.png", dpi=300)
plt.close(fig)

# Figure 3: stationary response hierarchy
pr, _, _ = fit_power(h, restricted)
pf, _, _ = fit_power(h, full)
fig = plt.figure(figsize=(6.4, 4.6))
ax = fig.add_subplot(111)
ax.loglog(h, restricted, label=fr"$\|\mathscr{{L}}Q_g\|$ ($p={pr:.4f}$)")
ax.loglog(h, full, label=fr"$\|\mathscr{{L}}\|$ ($p={pf:.4f}$)")
ax.set_xlabel(r"Refinement scale $h$")
ax.set_ylabel("Response norm")
ax.set_title("Restricted versus unrestricted stationary response")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figure3_stationary_response.pdf")
fig.savefig(OUT / "figure3_stationary_response.png", dpi=300)
plt.close(fig)

print("Kgg norm exponent, R2:", fit_power(h, gg[0])[0], fit_power(h, gg[0])[2])
print("Kgp norm exponent, R2:", fit_power(h, gp)[0], fit_power(h, gp)[2])
print("Kgg singular exponents:", [fit_power(h, gg[i])[0] for i in range(4)])
print("restricted response exponent:", fit_power(h, restricted)[0])
print("full response exponent:", fit_power(h, full)[0])
