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
NSIDE = 4096
PIX_SIZE = 1./np.sqrt(3.)/NSIDE

for idx in range(22):
	zmax_str    = f'{0.2*idx + 0.2:.1f}'
	zsource_str = f'{0.2*idx + 0.3:.1f}'

	kappa_map = hp.read_map(kappa_dir + 'total_zmax' + zmax_str + '_zsource' + zsource_str + '_kap.fits')

	kappa_map = hp.smoothing(kappa_map, sigma = PIX_SIZE)
	
	d, alpha = d_alpha_from_kappa(kappa_map)
	
	hp.write_map(f'output/for_lensing/smooth_1pix_kappa_shell_{idx+1}.fits',     kappa_map, overwrite = True)
	hp.write_map(f'output/for_lensing/smooth_1pix_d_shell_{idx+1}.fits',     d, overwrite = True)
	hp.write_map(f'output/for_lensing/smooth_1pix_alpha_shell_{idx+1}.fits', alpha, overwrite = True)
