from scipy.interpolate import interp1d
import numpy as np

"""
Cosmological parameters
"""
omegab = 0.049
omegac = 0.261
omegam = omegab + omegac
h      = 0.68
H0     = 100*h
ns     = 0.965
sigma8 = 0.81

"""
Hubble as a function of redshift
"""
H      = lambda z: H0*np.sqrt(omegam*(1+z)**3+1-omegam)

"""
Speed of light
"""
c = 3e5

"""
Redshift slices
"""
nz = 100000
z1 = 0.0
z2 = 6.0
za = np.linspace(z1,z2,nz)
dz = za[1]-za[0]

"""
Convert distance (in Mpc) to redshift
"""
dchidz = lambda z: c/H(z)
chi = np.cumsum(dchidz(za))*dz
z_from_chi = interp1d(chi,za)
