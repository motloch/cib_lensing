import numpy as np
from numpy import sin, cos
from astropy.io import fits
import healpy as hp

def assert_close(arr1, arr2, eps):
	"""
	Compare arrays are the same up to same precision
	"""
	assert(np.max(np.abs(arr1 - arr2)) < eps)

def d_alpha_from_kappa(kappa_map):
	"""
	Given a healpix convergence map kappa, return d, alpha as in the appendix of
	https://arxiv.org/pdf/astro-ph/0502469.pdf

	First calculating the deflection potential and then its gradient. d is the amplitude
	of the gradient and alpha is its deviation from the direction of the e_theta unit
	vector on the sphere.

	Input:  Healpix map of resolution NSIDE
	Return: Two healpix maps of the same resolution
	"""

	nside = hp.get_nside(kappa_map)
	lmax = 3*nside - 1

	#Go to spherical harmonics
	kappa_lm = hp.map2alm(kappa_map)

	#Convert to potential
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
	Given arrays of angular positions of points th, ph and maps of amplitude and direction
	of the gradient of the deflection potential, find the values of th, ph after deflection
	"""
	assert(len(th) == len(ph))
	nside = hp.get_nside(d_map)
	assert(hp.get_nside(alpha_map) == nside)

	#Which pixels we are actually probing, values of d and alpha at these points
	pix = hp.ang2pix(nside, th, ph)
	d     = d_map[pix]
	alpha = alpha_map[pix]

	cos_th = cos(th)	
	sin_th = sin(th)
	cos_ph = cos(ph)	
	sin_ph = sin(ph)
	cos_alpha = cos(alpha)
	sin_alpha = sin(alpha)
	cos_d = cos(d)
	sin_d = sin(d)

	#Note that because we want to go from the unlensed positions to the lensed positions,
	#we have to flip the sign of "d" from what is in https://arxiv.org/pdf/astro-ph/0502469.pdf
	xyz = np.zeros((len(th), 3))
	xyz[:,0] = - cos_alpha*cos_ph*cos_th*sin_d + sin_alpha*sin_d*sin_ph + cos_d*cos_ph*sin_th
	xyz[:,1] = - cos_alpha*sin_ph*cos_th*sin_d - sin_alpha*sin_d*cos_ph + cos_d*sin_ph*sin_th
	xyz[:,2] = cos_d*cos_th + cos_alpha*sin_d*sin_th

	return hp.vec2ang(xyz)

def get_lensed_pixel_numbers_from_xyz(x, y, z, d_map, alpha_map):
			"""
			Given the unlensed 3D position, return healpy pixel numbers of the corresponding
			lensed positions. The pixel numbers are in the resolutions of the two lensing maps.
			"""
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
			pix_len = hp.ang2pix(nside, th_len, ph_len)

			return pix_len
