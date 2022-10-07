"""
Calculate lensed CIB maps. Analogous to the unlensed code, only we process one redshift
bin at the time to avoid having to load the kappa, d, alpha maps all the time (or storing
all of them in memory).

In this code, we magnify using 1/(1-kappa)^2, which should be the proper way.
"""
import healpy as hp
import h5py
from   cosmology import *
import numpy as np
import time
import sys
from utils import get_lensed_pixel_numbers_from_xyz

NSIDE = 4096			#Healpix NSIDE determining resolution of the map
BATCHSIZE = 10000000	#How many haloes we process simultaneously
N_Z_SHELLS  = 23		#How many redshift bins we want
DZ = 0.2				#Size of the redshift bin

NSIDE = 4096
PIX_SIZE = 1./np.sqrt(3.)/NSIDE

"""
Process input parameters
"""

try:
  assert(len(sys.argv) == 3)
except:
	print('Wrong number of arguments inputted')
	print('Requested: frequency redshift_bin')
	print('Example: 3 1')
	exit()

FREQ_IDX = int(sys.argv[1]) #Flux at which frequency we want to use? Given by the zero-based 
														#    index in the frequency array
Z_IDX		 = int(sys.argv[2]) #Which redshift bin do we want to use?

assert(Z_IDX > 0)
assert(Z_IDX < N_Z_SHELLS)
ZMIN = Z_IDX*DZ
ZMAX = ZMIN + DZ

"""
Load the catalog
"""
dir_path = '/scratch/r/rbond/gstein/peak-patch-runs/current/jason_cib_lensing/2020/cib_mapmaking/galaxy_catalogue/'
galcat = h5py.File(dir_path + 'galaxy_catalogue_latest.h5', 'r')

"""
Print descriptive information
"""

print("fields included in the file are:")
print(galcat.keys())
print("shapes of these fields are:")
print([galcat[key].shape for key in galcat.keys()])

print('columns in flux arrays are for frequencies:')
print(galcat['observation_frequencies'][:])
print('chosen frequency:')
chosen_freq = galcat['observation_frequencies'][FREQ_IDX]
print(chosen_freq)

print("number of central galaxies:")
print(galcat['mass_cen'].shape[0])

print("number of the satellite galaxies:")
print(galcat['mass_sat'].shape[0])

nsats = galcat['nsat_inhalo']
print("Number of satellites in individual haloes is:")
print(nsats)
print("Minimal number of satellites:")
print(np.min(nsats))
print("Maximal number of satellites:")
print(np.max(nsats))

"""
Healpix maps to store the results - CIB fluxes at chosen frequency, from galaxies
at various redshift slices
"""

cib_map   = np.zeros( hp.nside2npix(NSIDE) )
time_at_start = time.time()

"""
Lensing fields
"""

kappa_dir = '/scratch/r/rbond/jasonlee/cib_lensing2020/kappa_maps/total/'
zmax_str    = f'{ZMIN:.1f}'
zsource_str = f'{ZMIN + 0.1:.1f}'

kappa_map = hp.read_map(kappa_dir + 'total_zmax' + zmax_str + '_zsource' + zsource_str + '_kap.fits')
kappa_map = hp.smoothing(kappa_map, sigma = PIX_SIZE)

d_map			= hp.read_map(f'output/for_lensing/smooth_1pix_d_shell_{Z_IDX}.fits')
alpha_map = hp.read_map(f'output/for_lensing/smooth_1pix_alpha_shell_{Z_IDX}.fits')

try:
	assert(len(kappa_map) == 12*NSIDE**2)
	assert(len(d_map)     == 12*NSIDE**2)
	assert(len(alpha_map) == 12*NSIDE**2)
except:
	print('One of the maps (kappa, d, alpha) has wrong NSIDE. Aborting.')
	exit()

"""
Function that loops over the galaxies and adds their fluxes to the CIB maps. Takes a
single parameter, determining whether we are summing over centrals 'cen' or satellites
'sat'
"""

def add_cib_fluxes(which):
	
	if which not in ['cen', 'sat']:
		print("add_cib_fluxes needs either 'cen' or 'sat' as an argument!")
		exit()

	num_batches = int(np.ceil(len(galcat['xpos_' + which])/BATCHSIZE))

	#Go over all galaxies in batches
	for batch_idx in np.arange(num_batches): 

		#Which galaxies we go over in the current batch.
		idx_st  = BATCHSIZE * batch_idx
		idx_end = BATCHSIZE * (batch_idx+1)

		#Their positions, fluxes and redshifts
		x = galcat['xpos_' + which][ idx_st : idx_end ]
		y = galcat['ypos_' + which][ idx_st : idx_end ]
		z = galcat['zpos_' + which][ idx_st : idx_end ]
		
		flux = np.array([
						flux[FREQ_IDX] 
						for 
						flux in galcat['flux_' + which][ idx_st : idx_end ]
					])

		chi = np.sqrt(x**2+y**2+z**2)
		redshift = zofchi(chi)

		#For each galaxy determine whether it is in the given redshift bin
		is_in_z_bin = (redshift >= ZMIN) * (redshift < ZMAX)

		#Convert 3D unlensed positions to healpix pixels
		pix_unl = hp.vec2pix(
								NSIDE, 
								x[is_in_z_bin], 
								y[is_in_z_bin],
								z[is_in_z_bin]
							)

		#Do the lensing operation and convert lensed positions to healpix pixels
		pix_len = get_lensed_pixel_numbers_from_xyz(
														x[is_in_z_bin],
														y[is_in_z_bin],
														z[is_in_z_bin],
														d_map,
														alpha_map
													)

		#Add fluxes to the map at appropriate pixels
		np.add.at(
						cib_map,
						pix_len,
						#Include magnification of the source due to lensing
						#https://microlensing-source.org/tutorial/magnification/
						#http://articles.adsabs.harvard.edu/pdf/1964MNRAS.128..295R
						flux[is_in_z_bin]/(1 - kappa_map[pix_unl])**2
					)

		#Check execution
		if batch_idx % 5 == 0:

			print('Timing information')
			print(batch_idx, which, time.time() - time_at_start)
			print(idx_st, idx_end)

"""
Run the calculation
"""
print('Adding CIB fluxes of central galaxies')
add_cib_fluxes('cen')
print('Adding CIB fluxes of satellite galaxies')
add_cib_fluxes('sat')

"""
Normalize, print summary information and save
"""

cib_map = cib_map * (12*NSIDE**2) / (4*np.pi)
print(
"Mean, variance, and max of shell %d:" % (Z_IDX), 
np.mean(cib_map),
np.var( cib_map), 
np.max( cib_map)
)
hp.write_map('output/lensed_proper_magnification/smooth_1pix_cib_' + str(chosen_freq) + '_shell_%d.fits' % (Z_IDX), 
	cib_map,
	overwrite = True
)
