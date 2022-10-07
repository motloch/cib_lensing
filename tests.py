"""
Test our routines
"""
import numpy as np
from numpy import sin, cos, sqrt, pi
from astropy.io import fits
import healpy as hp
import matplotlib.style
import matplotlib as mpl
mpl.use('agg')
mpl.style.use('classic')
import matplotlib.pyplot as plt
from utils import d_alpha_from_kappa, lens_th_ph, assert_close
from utils import get_lensed_pixel_numbers_from_xyz

"""
Test deflection angle components d (amplitude) and alpha (deviation) for a simple case of
kappa = 0.01 * cos(theta)
"""

NSIDE = 512
A = 0.01

th, ph = hp.pix2ang(NSIDE, range(12*NSIDE**2))

kappa_map = A*cos(th)

d, alpha = d_alpha_from_kappa(kappa_map)

try:
	d_should_be = A*sin(th)
	assert(np.max(d - d_should_be) < 1e-8)

	alpha_should_be = pi

	#Do this to get around the fact that pi and -pi are identical and sometimes numerical
	#fluctuation can wrap us around
	assert(np.max(
			(sin(alpha) - sin(alpha_should_be))**2 + 
			(cos(alpha) - cos(alpha_should_be))**2
			)
			< 1e-5
		)
	print('All tests passed')
except:
	print('Something went wrong in test 1')

"""
Test the actual deflection code on a random sample of points
"""

TH = np.array([0.4, 1.4, 2.4])
PH = np.array([2.3, 1.2, 2.2])

th_lensed, ph_lensed = lens_th_ph(TH, PH, d, alpha)

th_lensed_should_be = np.array([0.403894, 1.40985, 2.40675])
ph_lensed_should_be = np.array([2.3, 1.2, 2.2])

try:
	assert_close(th_lensed, th_lensed_should_be, 2e-5)
	assert_close(ph_lensed, ph_lensed_should_be, 1e-6)
	print('All tests passed')
except:
	print('Something went wrong in test 2')

"""
Test deflection angle components d (amplitude) and alpha (deviation) for a simple case of
kappa = 0.01 * cos(theta) rotated by 45 degrees
"""

NSIDE = 512
A = 0.01

th, ph = hp.pix2ang(NSIDE, range(12*NSIDE**2))

kappa_map = A/sqrt(2.)*(cos(th) + sin(th)*cos(ph))

d, alpha = d_alpha_from_kappa(kappa_map)

try:
	d_should_be = A*sqrt((sin(ph)**2 + (sin(th) - cos(th)*cos(ph))**2)/2)
	assert(np.max(d - d_should_be) < 1e-7)

	alpha_should_be = np.arctan2(-sin(ph), -sin(th) + cos(th)*cos(ph))

	#Do this to get around the fact that pi and -pi are identical and sometimes numerical
	#fluctuation can wrap us around
	assert(np.max(
			(sin(alpha) - sin(alpha_should_be))**2 + 
			(cos(alpha) - cos(alpha_should_be))**2
			)
			< 1e-9
		)

	print('All tests passed')
except:
	print('Something went wrong in test 3')

"""
Test the actual deflection code on a random sample of points - rotated magnification
"""

TH = np.array([0.4, 1.4, 2.4])
PH = np.array([2.3, 1.2, 2.2])

th_lensed, ph_lensed = lens_th_ph(TH, PH, d, alpha)

th_lensed_should_be = np.array([0.407125, 1.40654, 2.40169])
ph_lensed_should_be = np.array([2.31332, 1.20668, 2.20848])

try:
	assert_close(th_lensed, th_lensed_should_be, 2e-5)
	assert_close(ph_lensed, ph_lensed_should_be, 1e-5)
	print('All tests passed')
except:
	print('Something went wrong in test 4')

"""
Compare the unlensed maps with Jason
"""
jason_map = hp.read_map('jason/mapmaker/545GHz_shells/cib_shell_14.fits')
pavel_map = hp.read_map('output/unlensed/cib_545_shell_13.fits')
assert_close(jason_map, pavel_map, 1e-5)

"""
Check the shells give different results
"""
wrong_map = hp.read_map('output/unlensed/cib_545_shell_15.fits')
assert(np.sum(jason_map[:100]**2) > 0)
assert(np.sum(jason_map[:100]**2) != np.sum(wrong_map[:100]**2))

"""
Check d, alpha look sensible - image output
"""
d = hp.read_map('output/for_lensing/d_shell_8.fits')
hp.mollview(d)
plt.savefig('output/tests/d_shell_8.png')

alpha = hp.read_map('output/for_lensing/alpha_shell_8.fits')
hp.mollview(alpha)
plt.savefig('output/tests/alpha_shell_8.png')

"""
Compare Pavel's lensing and Jason's
"""
unlensed_map     = hp.read_map('output/unlensed/cib_545_shell_1.fits')
pavel_lensed_map = hp.read_map('output/lensed/cib_545_shell_1.fits')
jason_lensed_dir = '/scratch/r/rbond/jasonlee/cib_lensing2020/maps/lensed2020/Nov2020/545GHz/nside4096/'
jason_lensed_map = hp.read_map(jason_lensed_dir + 'lensed_zmin0.2_zmax0.4.fits')

hp.gnomview(unlensed_map, xsize = 100, min = -0.05, max = 0.45)
plt.savefig('output/tests/sample_unl_shell_1.png')
hp.gnomview(pavel_lensed_map, xsize = 100, min = -0.05, max = 0.45)
plt.savefig('output/tests/sample_pavel_lensed_shell_1.png')
hp.gnomview(jason_lensed_map, xsize = 100, min = -0.05, max = 0.45)
plt.savefig('output/tests/sample_jason_lensed_shell_1.png')

"""
Compare Pavel's lensing and Jason's at the point of maximal deflection
"""
d = hp.read_map('output/for_lensing/d_shell_1.fits')
alpha = hp.read_map('output/for_lensing/alpha_shell_1.fits')
max_deflection_at = hp.pix2ang(hp.get_nside(d), np.argmax(d), lonlat = True)

hp.gnomview(d, xsize = 100, rot = max_deflection_at)
plt.savefig('output/tests/sample_B_d_shell_1.png')
hp.gnomview(alpha, xsize = 100, rot = max_deflection_at)
plt.savefig('output/tests/sample_B_alpha_shell_1.png')
hp.gnomview(unlensed_map, xsize = 100, min = -0.05, max = 0.45, rot = max_deflection_at)
plt.savefig('output/tests/sample_B_unl_shell_1.png')
hp.gnomview(pavel_lensed_map, xsize = 100, min = -0.05, max = 0.45, rot = max_deflection_at)
plt.savefig('output/tests/sample_B_pavel_lensed_shell_1.png')
hp.gnomview(jason_lensed_map, xsize = 100, min = -0.05, max = 0.45, rot = max_deflection_at)
plt.savefig('output/tests/sample_B_jason_lensed_shell_1.png')

"""
Check that the main code that combines lensing and searching for the appropriate healpix
pixel works as expected
"""
#Our new code
th_unlensed = np.array([0.40, 1.40, 2.40])
ph_unlensed = np.array([2.3, 1.2, 2.2])

x = cos(ph_unlensed)*sin(th_unlensed)
y = sin(ph_unlensed)*sin(th_unlensed)
z = cos(th_unlensed)

NSIDE = 512
A = 0.01
th, ph = hp.pix2ang(NSIDE, range(12*NSIDE**2))
kappa_map = A*cos(th)
d, alpha = d_alpha_from_kappa(kappa_map)

our_results = get_lensed_pixel_numbers_from_xyz(x, y, z, d, alpha)

#We know the correct answer - where does it land?
th_lensed = np.array([0.403894, 1.40985, 2.40675])
ph_lensed = np.array([2.3, 1.2, 2.2])

try:
	for idx in range(len(th_unlensed)):
		#Contributions to the scalar product between the currently investigated position and all
		#the Healpix pixels
		x_contribution = cos(ph_lensed[idx])*sin(th_lensed[idx])*cos(ph)*sin(th)
		y_contribution = sin(ph_lensed[idx])*sin(th_lensed[idx])*sin(ph)*sin(th)
		z_contribution = cos(th_lensed[idx])*cos(th)
		distances = x_contribution + y_contribution + z_contribution

		#Check we found the correct pixel in two different ways
		assert(our_results[idx] == np.argmax(distances))
		assert(our_results[idx] == hp.ang2pix(NSIDE, th_lensed[idx], ph_lensed[idx]))
		#Check that displacements are not so small we stay in a single pixel
		assert(our_results[idx] != hp.ang2pix(NSIDE, th_unlensed[idx], ph_unlensed[idx]))

	print('All tests passed')
except:
	print('Something went wrong in test 5')

"""
Check that the total flux in the map is about 4 MJy, as George says (based on our notes it
should be 4.18 MJy, though here it seems like that was based on an old simulation)
"""
s = 0
for idx in range(23):
	s += np.sum(
					hp.read_map('output/unlensed/cib_545_shell_' + str(idx) + '.fits')
				)
pixel_size = 4*np.pi/12./4096**2
total_flux = s*pixel_size

assert_close(total_flux, 4., 0.2)
