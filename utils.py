import numpy as np
from numpy import sin, cos
from astropy.io import fits
import healpy as hp

def assert_close(arr1, arr2, eps):
	"""
	Compare arrays are the same up to the defined precision
	"""
	assert(np.max(np.abs(arr1 - arr2)) < eps)

def d_alpha_from_kappa(kappa_map):
	"""
	Given a healpix convergence map kappa, return the deflection magnitude (d) and direction
	(alpha) fields, as defined in the appendix of https://arxiv.org/pdf/astro-ph/0502469.pdf

	First calculate the deflection potential, then its gradient. d is the amplitude
	of the gradient and alpha is its deviation from the direction of the e_theta unit
	vector on the sphere.

	Inputs:	
		kappa_map - Healpix map of resolution NSIDE representing the convergence field
	Returns: 
		Two healpix maps of the same resolution representing d and alpha fields
	"""

	nside = hp.get_nside(kappa_map)
	lmax = 3*nside - 1

	#Go to spherical harmonics
	kappa_lm = hp.map2alm(kappa_map)

	#Convert to gravitational potential
	l = np.arange(lmax+1)
	l[0] = 1. #To avoid division by zero, will be corrected shortly
	to_phi_coefficients = 2./l/(l+1.)
	to_phi_coefficients[0] = 0

	phi_lm = hp.almxfl(kappa_lm, to_phi_coefficients)

	#Calculate derivatives
	m, d_theta, d_phi = hp.alm2map_der1(phi_lm, nside)

	#The amplitude and direction of the gradient
	d = np.sqrt(d_theta**2 + d_phi**2)
	alpha = np.arctan2(d_phi, d_theta)

	return d, alpha

def lens_th_ph(th, ph, d_map, alpha_map):
	"""
	Given 
		* arrays of angular positions of points th, ph and 
		* maps of amplitude (d_map) and direction (alpha_map) of the gradient of the deflection potential
	finds the values of th, ph after deflection

	Inputs:  
		th, ph - numpy arrays of the same length representing angular coordinates
							of the unlensed positions
		d_map, alpha_map - healpix arrays of the same nside representing the deflection
							amplitude and direction
	Returns: 
		numpy arrays of the length len(th) representing angular coordinates
							of the lensed positions
	"""
	assert(len(th) == len(ph))
	nside = hp.get_nside(d_map)
	assert(hp.get_nside(alpha_map) == nside)

	# In which pixels do the investigated points lie? What are values of d, alpha there?
	pix = hp.ang2pix(nside, th, ph)
	d     = d_map[pix]
	alpha = alpha_map[pix]

	# For speedup
	cos_th = cos(th)	
	sin_th = sin(th)
	cos_ph = cos(ph)	
	sin_ph = sin(ph)
	cos_alpha = cos(alpha)
	sin_alpha = sin(alpha)
	cos_d = cos(d)
	sin_d = sin(d)

	# Note that because we want to go from the unlensed positions to the lensed positions,
	# we have to flip the sign of "d" in Eq. A15, A16 of https://arxiv.org/pdf/astro-ph/0502469.pdf
	xyz = np.zeros((len(th), 3))
	xyz[:,0] = - cos_alpha*cos_ph*cos_th*sin_d + sin_alpha*sin_d*sin_ph + cos_d*cos_ph*sin_th
	xyz[:,1] = - cos_alpha*sin_ph*cos_th*sin_d - sin_alpha*sin_d*cos_ph + cos_d*sin_ph*sin_th
	xyz[:,2] = cos_d*cos_th + cos_alpha*sin_d*sin_th

	return hp.vec2ang(xyz)

def get_lensed_pixel_numbers_from_xyz(x, y, z, d_map, alpha_map):
			"""
			Given the unlensed 3D positions, return "healpix pixel numbers" of the corresponding
			lensed positions. The pixel numbers are in the resolutions of the two lensing maps.

			Inputs:  
				x, y, z - numpy arrays of the same length representing x, y, z coordinates
									of the unlensed positions
				d_map, alpha_map - healpix arrays of the same nside representing the deflection
									amplitude and direction
			Returns: 
				numpy array of length len(x) representing "healpix pixel numbers" of the
									pixels in which we can find (x,y,z) after lensing
			"""
			assert(len(d_map) == len(alph_map))
			assert(len(x) == len(y) == len(z))

			#Angular position before deflection
			th_unl, ph_unl = hp.vec2ang(
													np.transpose(
														np.array(
															[x, y, z]
														)
													)
												)

			#Angular position after deflection
			th_len, ph_len = lens_th_ph(th_unl, ph_unl, d_map, alpha_map)

			#Convert angular positions to healpix pixels
			nside = hp.get_nside(alpha_map)
			pixel_numbers_len = hp.ang2pix(nside, th_len, ph_len)

			return pixel_numbers_len
