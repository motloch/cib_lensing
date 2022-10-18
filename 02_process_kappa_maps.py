"""
Given gravitational lensing convergence maps from a simulation, calculate and save the
deflection magnitude (d) and direction (alpha) fields. They are defined in the appendix of
https://arxiv.org/pdf/astro-ph/0502469.pdf
"""
import numpy as np
from astropy.io import fits
import healpy as hp
import matplotlib.style
import matplotlib as mpl
mpl.use('agg')
mpl.style.use('classic')
import matplotlib.pyplot as plt
from utils import d_alpha_from_kappa

kappa_dir = '/scratch/r/rbond/jasonlee/cib_lensing2020/kappa_maps/total/'

NSIDE = 4096										#Healpix NSIDE determining resolution of the map
PIX_SIZE = 1./np.sqrt(3.)/NSIDE #Effective diameter of a pixel, treating it as a circle
N_Z_SHELLS  = 23								#How many redshift bins we want

# Cycle over the redshift shells (the closest one to us has no lensing)
for z_idx in range(1, N_Z_SHELLS):

	# Redshift of the source, maximal redshift of the lenses
	zsource_str = f'{0.2*z_idx + 0.1:.1f}'
	zmax_str    = f'{0.2*z_idx      :.1f}'

	# Load convergence map
	kappa_map = hp.read_map(kappa_dir + 'total_zmax' + zmax_str + '_zsource' + zsource_str + '_kap.fits')

	# Smooth it at the pixel resolution
	kappa_map = hp.smoothing(kappa_map, sigma = PIX_SIZE)
	
	# Calculate deflection amplitude and direction
	d, alpha = d_alpha_from_kappa(kappa_map)
	
	# Save the results
	hp.write_map(f'output/for_lensing/smooth_1pix_kappa_shell_{z_idx}.fits', kappa_map, overwrite = True)
	hp.write_map(f'output/for_lensing/smooth_1pix_d_shell_{z_idx}.fits',     d,					overwrite = True)
	hp.write_map(f'output/for_lensing/smooth_1pix_alpha_shell_{z_idx}.fits', alpha,			overwrite = True)
