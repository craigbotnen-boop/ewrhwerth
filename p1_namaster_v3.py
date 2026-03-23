"""
ACT DR6 kappa x DESI/eBOSS delta  (pseudo-Cl)  v3
Includes all bug-fixes from the 8-point audit.
"""

# ---------- imports ----------
import os, glob, json, yaml, hashlib, numpy as np
import healpy as hp, pymaster as nmt
from tqdm import tqdm
from astropy.table import Table
from scipy.stats import chi2 as chi2dist          # moved to top

# ---------- configuration ----------
DATA = "/content/drive/MyDrive/P1_Data/"
OUT  = f"{DATA}/outputs/"
NSIDE = 1024
BIN_W = 40
N_SIMS = 100          # set 400+ on HPC

os.makedirs(OUT, exist_ok=True)

def one(match):                           # auto-finder
    hit = glob.glob(DATA + match)
    if not hit:
        raise FileNotFoundError(f"Missing file *{match}*")
    return hit[0]

kappa_alm = one("*kappa*alm*.fits")
curl_alm  = one("*curl*alm*.fits")
mask_fits = one("*mask*4096*.fits")
gal_cat   = one("*clustering*.fits")
rand_cat  = one("*random*.fits")
sim_dir   = one("*kappa_sims*/")          # folder of kappa simulations


# ---------- SHA-256 helper ----------
def sha256_chunked(path, chunk_size=65536):
    """Chunked SHA-256 -- safe for multi-GB files."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


# ---------- robust ACT-alm reader ----------
def read_act_alm(path):
    try:
        alm = hp.read_alm(path)
        alm[np.isnan(alm)] = 0.0
        return alm
    except (ValueError, OSError):
        import fitsio, math
        tbl = fitsio.FITS(path)[1].read()
        idx, real, imag = tbl['index'], tbl['REAL'], tbl['IMAG']
        lmax = int((-1 + np.sqrt(1 + 8*int(idx.max()))) / 2)    # exact inverse
        alm  = np.zeros(hp.Alm.getsize(lmax), np.complex128)
        alm[idx] = real + 1j*imag
        alm[np.isnan(alm)] = 0.0
        return alm                                            # no almxfl misuse


# ---------- maps ----------
kap_map  = hp.alm2map(read_act_alm(kappa_alm), NSIDE)
curl_map = hp.alm2map(read_act_alm(curl_alm),  NSIDE)

mask_act = hp.ud_grade(hp.read_map(mask_fits), NSIDE)
mask_act = (mask_act > 0.5).astype(float)                     # binary first
mask_act = nmt.mask_apodization(mask_act, 2.0, "C2")


# ---------- galaxy delta-map with weights ----------
def make_delta(cat_f, rand_f, nside):
    gal  = Table.read(cat_f)
    rnd  = Table.read(rand_f)

    w_g = (gal['WEIGHT_SYS'] * gal['WEIGHT_COMP'] * gal['WEIGHT_ZFAIL']).astype(float)
    w_r = rnd['WEIGHT'].astype(float) if 'WEIGHT' in rnd.colnames else np.ones(len(rnd))

    npix = 12 * nside**2
    pix_g = hp.ang2pix(nside, gal['RA'],  gal['DEC'], lonlat=True)
    pix_r = hp.ang2pix(nside, rnd['RA'],  rnd['DEC'], lonlat=True)

    n_g = np.bincount(pix_g, weights=w_g, minlength=npix)
    n_r = np.bincount(pix_r, weights=w_r, minlength=npix)
    alpha = n_g.sum() / n_r.sum()

    delta = np.zeros(npix)
    sel   = n_r > 0
    delta[sel] = (n_g[sel] - alpha*n_r[sel]) / (alpha*n_r[sel])

    gal_mask = np.zeros(npix)
    gal_mask[sel] = 1.0
    gal_mask = nmt.mask_apodization(gal_mask, 1.0, "C2")

    inside = gal_mask > 0
    delta[inside] -= delta[inside].mean()
    delta *= gal_mask
    return delta, gal_mask

delta_map, mask_gal = make_delta(gal_cat, rand_cat, NSIDE)


# ---------- NaMaster fields & workspaces ----------
bins = nmt.NmtBin.from_nside_linear(NSIDE, BIN_W)

f_gal   = nmt.NmtField(mask_gal, [delta_map])
f_kappa = nmt.NmtField(mask_act, [kap_map])
f_curl  = nmt.NmtField(mask_act, [curl_map])

w_kg = nmt.NmtWorkspace(); w_kg.compute_coupling_matrix(f_kappa, f_gal, bins)
w_cg = nmt.NmtWorkspace(); w_cg.compute_coupling_matrix(f_curl,  f_gal, bins)

cl_kg = w_kg.decouple_cell(nmt.compute_coupled_cell(f_kappa, f_gal))[0]
cl_cg = w_cg.decouple_cell(nmt.compute_coupled_cell(f_curl,  f_gal))[0]
ells  = bins.get_effective_ells()
windows = w_kg.get_bandpower_windows()


# ---------- kappa-simulation error bars ----------
sim_list = sorted(glob.glob(sim_dir + "*alm*.fits"))[:N_SIMS]
cl_sims = []
for p in tqdm(sim_list, desc="kappa-sim cross"):
    sim_map = hp.alm2map(read_act_alm(p), NSIDE)
    f_sim   = nmt.NmtField(mask_act, [sim_map])
    cl_sims.append(
        w_kg.decouple_cell(nmt.compute_coupled_cell(f_sim, f_gal))[0]
    )
cl_sims = np.asarray(cl_sims)
err = cl_sims.std(axis=0)


# ---------- curl null chi-squared ----------
finite = err > 0
chi2  = np.sum((cl_cg[finite]/err[finite])**2)
dof   = int(finite.sum())
pte   = 1 - chi2dist.cdf(chi2, dof)


# ---------- outputs ----------
np.savez(f"{OUT}/p1_cross.npz", ell=ells, cl=cl_kg, err=err)
np.savez(f"{OUT}/p1_curl_null.npz", ell=ells, cl=cl_cg, err=err)
np.savez(f"{OUT}/bandpower_windows.npz", bpw=windows)

manifest = dict(
    files={os.path.basename(p): sha256 for p, sha256 in
           [(kappa_alm, sha256_chunked(kappa_alm)),
            (curl_alm,  sha256_chunked(curl_alm)),
            (mask_fits, sha256_chunked(mask_fits)),
            (gal_cat,   sha256_chunked(gal_cat)),
            (rand_cat,  sha256_chunked(rand_cat))]},
    nside=NSIDE, bin_width=BIN_W, n_sims=N_SIMS,
    chi2_curl_null=float(chi2), dof=dof, pte=float(pte)
)
yaml.safe_dump(manifest, open(f"{OUT}/manifest.yaml", "w"))

print("DONE  Spectra + cov saved to", OUT)
print("ell[0:5]      ", ells[:5])
print("Cl_kg[0:5]    ", cl_kg[:5])
print("sigma[0:5]    ", err[:5])
print(f"Curl-null chi2 = {chi2:.2f}   PTE = {pte:.3f}")
