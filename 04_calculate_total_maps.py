"""
Sum CIB fluxes from individual redshift shells to create the total CIB map
"""
import healpy as hp
import numpy as np

def get_total_cib_map(which, frequency):
	"""
	Add CIB fluxes from individual redshift shells. Allows us to pick which set of maps
	(unlensed, lensed, ...) we want and which frequency.
	"""
	for idx in range(23):
		if idx == 0:
			cib_map = hp.read_map('output/' + which + '/smooth_1pix_cib_' + frequency + '_shell_' + str(idx) + '.fits')
		else:
			cib_map += hp.read_map('output/' + which + '/smooth_1pix_cib_' + frequency + '_shell_' + str(idx) + '.fits')

	hp.write_map(
			'output/' + which + '/smooth_1pix_cib_' + frequency + '_total.fits', 
			cib_map,
			overwrite = True
		)

get_total_cib_map('lensed_proper_magnification', '545')
get_total_cib_map('lensed_proper_magnification', '353')
get_total_cib_map('lensed_proper_magnification', '217')
