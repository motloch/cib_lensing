"""
Given a catalog of galaxy angular positions, redshifts and fluxes, calculate
unlensed CIB maps. Outputs both the total CIB flux and a series of fluxes from galaxies
located in several redshift shells [z, z+dz]. 

Loops over galaxies, finds their positions, adds fluxes to the corresponding pixels. 

Mostly refactoring Jaemyoung (Jason) Lee's code, only made sure centrals and satellites
are treated in uniform manner. 
"""
import healpy as hp
import h5py
from   cosmology import *
import numpy as np
import time
import sys

NSIDE = 4096					#Healpix NSIDE determining resolution of the map
BATCHSIZE = 10000000	#How many haloes we process simultaneously
N_Z_SHELLS  = 23			#How many redshift bins we want
DZ = 0.2							#Size of each redshift bin

"""
Load the requested frequency from the run arguments
"""

try:
  assert(len(sys.argv) == 2)
except:
	print('Wrong number of arguments inputted')
	print('Requested: frequency index, as defined in the input file')
	print('Example: 3')
	exit()

FREQ_IDX = int(sys.argv[1]) #Flux at which frequency we want to use? Given by the zero-based 
														#index in the frequency array, as given in the input file

"""
Load the catalog
"""
dir_path = '/scratch/r/rbond/jasonlee/cib_lensing2020/gal_catalog/mapmaker/maps_vNov2021/checks/galaxy_catalogue/'
galcat = h5py.File(dir_path + 'galaxy_catalogue_latest.h5', 'r')

"""
Print descriptive information about the catalog
"""

print("Fields included in the file are:")
print(galcat.keys())
print("Shapes of these fields are:")
print([galcat[key].shape for key in galcat.keys()])

print('Columns in flux arrays are for frequencies:')
print(galcat['observation_frequencies'][:])
print('Chosen frequency:')
chosen_freq = galcat['observation_frequencies'][FREQ_IDX]
print(chosen_freq)

print("Number of central galaxies:")
print(galcat['mass_cen'].shape[0])

print("Number of the satellite galaxies:")
print(galcat['mass_sat'].shape[0])

nsats = galcat['nsat_inhalo']
print("Number of satellites in individual haloes is:")
print(nsats)
print("Minimal number of satellites:")
print(np.min(nsats))
print("Maximal number of satellites:")
print(np.max(nsats))

"""
Healpix maps to store the results in - CIB fluxes (at chosen frequency) from galaxies in
various redshift slices 
"""

cib_maps   = np.zeros( (N_Z_SHELLS, hp.nside2npix(NSIDE)) )
time_at_start = time.time()

def add_cib_fluxes(which):
	"""
	Function that loops over the galaxies and adds their fluxes to the CIB maps. Takes a
	single parameter, determining whether we are summing over centrals 'cen' or satellites
	'sat'. Does not return anything, alters the "cib_maps" array.
	"""
	
	if which not in ['cen', 'sat']:
		print("add_cib_fluxes needs either 'cen' or 'sat' as an argument!")
		exit()

	#Go over all galaxies in batches of predetermined size
	num_batches = int(np.ceil(len(galcat['xpos_' + which])/BATCHSIZE))
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
		redshift = z_from_chi(chi)

		#Loop over redshift bins
		for z_idx in range(N_Z_SHELLS):

			#For each galaxy determine whether it is in the given redshift bin
			is_in_z_bin = (redshift >= z_idx * DZ) * (redshift < (z_idx+1) * DZ)

			#Convert 3D positions to healpix pixels
			pix = hp.vec2pix(
						NSIDE, 
						x[is_in_z_bin], 
						y[is_in_z_bin],
						z[is_in_z_bin]
					  )

			#Add fluxes to the map at appropriate pixels
			np.add.at(cib_maps[z_idx], pix, flux[is_in_z_bin])

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

for z_idx in range(N_Z_SHELLS):
    cib_maps[z_idx] = cib_maps[z_idx] * (12*NSIDE**2) / (4*np.pi)
    print(
		"Mean, variance, and max of shell %d:" % (z_idx), 
		np.mean(cib_maps[z_idx]),
		np.var( cib_maps[z_idx]), 
		np.max( cib_maps[z_idx])
	)
    hp.write_map('output/unlensed/cib_' + str(chosen_freq) + '_shell_%d.fits' % (z_idx), 
			cib_maps[z_idx],
			overwrite = True
		)

"""
Total CIB map obtained by summing over all the redshift shells
"""

sum_of_shells = np.sum(cib_maps, axis = 0)

print(
	'Mean, variance, min, and max of added map: ', 
	np.mean(sum_of_shells), 
	np.var( sum_of_shells), 
	np.min( sum_of_shells), 
	np.max( sum_of_shells)
)

hp.write_map(
	'output/unlensed/cib_' + str(chosen_freq) + '_total.fits', 
	sum_of_shells,
	overwrite = True
)
