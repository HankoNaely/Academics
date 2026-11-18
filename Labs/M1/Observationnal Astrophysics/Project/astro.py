# ===========================================
# IMPORT
# ===========================================
import glob
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from astropy import units as u
from astropy.visualization import ZScaleInterval
from astropy.wcs import FITSFixedWarning
from astropy.modeling import models
import ccdproc as ccdp
import astropy.io.fits as pf
import specutils
import specutils.fitting as spf
from specutils.manipulation import extract_region
from specutils import Spectrum1D, SpectralRegion

warnings.simplefilter('ignore', FITSFixedWarning)

# Get script directory 
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '4-alpy_2021', '2021-11-18')

# LOADING DATA
# ==================================================================================================
# Loading reduced spectrum
sci_s1d_gamma = pf.open(os.path.join(data_dir, 'reduced', 'gamma_Cas_3_fw_s1d.fit'))
sci_s1d_epsilon = pf.open(os.path.join(data_dir, 'reduced', 'epsilon_Cas_3_fw_s1d.fit'))

lam_gamma = np.array([x[0] for x in sci_s1d_gamma[1].data])
lam_epsilon = np.array([x[0] for x in sci_s1d_epsilon[1].data])

flux_gamma = np.array([x[1] for x in sci_s1d_gamma[1].data])
flux_epsilon = np.array([x[1] for x in sci_s1d_epsilon[1].data])

# Loading master flats
mflat_s1d_gamma = ccdp.CCDData.read(os.path.join(data_dir, 'reference', 'master_flat_gamma_Cas_s1d.fits'))
mflat_s1d_epsilon = ccdp.CCDData.read(os.path.join(data_dir, 'reference', 'master_flat_epsilon_Cas_s1d.fits'))

# Loading wavelength calibraiotn 
mwcal_s1d_gamma = ccdp.CCDData.read(os.path.join(data_dir, 'reference', 'master_wcal_gamma_Cas_s1d.fits'))
mwcal_s1d_epsilon = ccdp.CCDData.read(os.path.join(data_dir, 'reference', 'master_wcal_epsilon_Cas_s1d.fits'))
# ===================================================================================================


# NORMALISATION OF MASTER FLAT (keep only small-scale variations)
# ==================================================================================================
# Perform high-degree polynomial fit 
'''x_ccd = np.linspace(0, mflat_s1d_gamma.data.shape[0],  mflat_s1d_gamma.data.shape[0])

mflat_coeff_gamma = np.polyfit(x_ccd, mflat_s1d_gamma.data, deg=11)
mflat_coeff_epsilon = np.polyfit(x_ccd, mflat_s1d_epsilon.data, deg=11)

# Compute polynomial model
mflat_fit_gamma = np.poly1d(mflat_coeff_gamma)(x_ccd)
mflat_fit_epsilon = np.poly1d(mflat_coeff_epsilon)(x_ccd)

# Normalise the flat field w/ this model
mflat_n_s1d_gamma = mflat_s1d_gamma.data / mflat_fit_gamma
mflat_n_s1d_epsilon = mflat_s1d_epsilon.data / mflat_fit_epsilon'''
# ==================================================================================================

# PLOT FIT AND REDUCED
# ==================================================================================================
'''fig = plt.figure(figsize=(10,8))
#
ax1 = fig.add_subplot(211)
ax1.plot(mflat_s1d_gamma.data, c='tab:blue')
ax1.plot(mflat_s1d_epsilon.data, c='tab:red')
ax1.plot(mflat_fit_gamma, c='tab:orange')
ax1.plot(mflat_fit_epsilon, c='tab:green')
ax1.axvline(850, ls='--', c='k', lw=0.4)
ax1.set_title('Master flat field and polynomial fit of Epsilon and Gamma')

ax2 = fig.add_subplot(212)
ax2.plot(mflat_n_s1d_gamma, c='tab:blue', label='Gamma')
ax2.plot(mflat_n_s1d_epsilon, c='tab:red', label='Epsilon')
ax2.axvline(850, ls='--', c='k', lw=0.4)
ax2.set_title('Normalised master flat field of Epsilon and Gamma')
ax2.legend()
plt.show()'''
# ==================================================================================================

# WAVELENGTH CALIBRATION
# ==================================================================================================
# Convert wcal spectrum to a specutils.Spectrum1D instance
mwcal_s1d_gamma = specutils.Spectrum1D(mwcal_s1d_gamma.data*u.adu)
mwcal_s1d_epsilon = specutils.Spectrum1D(mwcal_s1d_epsilon.data*u.adu)

# === GAMMA ===
# Detection of spectral lines
lines_gamma = spf.find_lines_derivative(mwcal_s1d_gamma, flux_threshold=7.e4)
lines_gamma[lines_gamma['line_type'] == 'emission']
print("%d emission lines identified" % len(lines_gamma))
print(lines_gamma)

# PLot identified lines
plt.figure()
plt.plot(mwcal_s1d_gamma.flux)
for l in lines_gamma:
    plt.axvline(l['line_center_index'], ls='--', c='grey', lw=0.5)
plt.show()

# Fit each line w/ a gaussian model
wcal_amp_gamma = []
wcal_x_gamma = []
wcal_std_gamma = []
plt.figure(figsize=(10,6))
plt.plot(lam_gamma, flux_gamma*3.e5/np.mean(flux_gamma), c='k', lw=0.5)
plt.plot(mwcal_s1d_gamma.flux)
print('Amplitude   x0     sigma')
for l in lines_gamma:
    sub_region_gamma = SpectralRegion((l['line_center_index']-4)*u.pix, (l['line_center_index']+4)*u.pix)
    sub_spectrum_gamma = extract_region(mwcal_s1d_gamma, sub_region_gamma)
    g = spf.estimate_line_parameters(sub_spectrum_gamma, models.Gaussian1D())
    wcal_amp_gamma.append(g.amplitude.value)
    wcal_x_gamma.append(g.mean.value)
    wcal_std_gamma.append(g.stddev.value)
    xx = np.linspace((l['line_center_index']-5)*u.pix,(l['line_center_index']+5)*u.pix,22)
    plt.plot(xx, g(xx))
    print("%9.1f   %5.1f  %.2f"%(g.amplitude.value, g.mean.value, g.stddev.value))

plt.show()

