"""
Create the total CIB maps by summing CIB fluxes from sources located in different redshift
shells
"""
import healpy as hp
import numpy as np

NUM_Z_SHELLS = 23
FREQUENCIES = ['545', '353', '217']

def get_total_cib_map(which, frequency):
	"""
	Add CIB fluxes from individual redshift shells. Allows us to pick which set of maps
	(unlensed, lensed) we want and which frequency.
	"""

	# In our model, the closest shell is not lensed (as there is no matter that could lens)
	cib_map = hp.read_map('output/unlensed/cib_' + frequency + '_shell_0.fits')

	for idx in range(1,NUM_Z_SHELLS):
			cib_map += hp.read_map('output/' + which + '/cib_' + frequency + '_shell_' + str(idx) + '.fits')

	return cib_map

for frequency in FREQUENCIES:
	for which in ['lensed', 'unlensed']:
		cib_map = get_total_cib_map(which, frequency)
		hp.write_map(
				'output/' + which + '/cib_' + frequency + '_total.fits',
				cib_map,
				overwrite = True
			)
