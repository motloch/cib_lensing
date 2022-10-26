# CIB LENSING #

This is a series of codes that can be used to calculate lensed cosmic infrared background
(CIB) signal from the [Websky Extragalactic simulations](https://arxiv.org/pdf/2001.08787.pdf).
The code can be easily modified to perform lensing on any point source signal, regardless of its source redshift distribution.

Unlike the well known cosmic microwave background (CMB), CIB is a signal that is not
sufficiently smooth to be lensed using an interpolation scheme such as [Lenspix](https://github.com/cmbant/lenspix).
As an additional complication not present for the CMB lensing, galaxies sourcing CIB are
distributed over a large range of redshifts and the lensing can not be encoded using a
single convergence map.

To solve the first issue, we perform lensing at the level of sourcing galaxies and for
each such galaxy calculate its lensed position following the formalism presented in Appendix of
[paper](https://arxiv.org/pdf/astro-ph/0502469.pdf). Additionally, we modulate fluxes of the
sources by a convergence-dependent factor.

To better address the fact that CIB sources are spread over a range of redshifts, we
group the source galaxies into redshift shells of thickness $\Delta z = 0.2$ and for the
purposes of lensing assume all such galaxies are located at the middle redshift of the
shell and are lensed by all matter between the shell and the observer. This neglects lensing by the galaxies
located in the same redsfhit shell.

## Input ##

Overall, the input provided to the codes:
* a galaxy catalog in an h5 format with positions, redshifts and fluxes in various frequencies
* a set of Healpix maps representing the gravitational lensing convergence when
  considering matter up to a particular redshift shell

Details of how these files were generated can be found in the simulation paper above.

## Description of files ##

File | Description | Command line arguments
--- | --- | ---
01_calculate_unlensed.py | Calculates the unlensed CIB signal | Integer specifying which frequency to use. Given as an index (0, 1, ...)
02_process_kappa_maps.py | Calculates lensing deflection amplitudes and directions from the convergence maps | None
03_calculate_lensed.py | Calculates the lensed CIB signal from galaxies in individual redshift shells | Integer specifying which frequency to use. Given as an index (0, 1, ...)
| | | Integer specifying which redshift shell to consider
04_calculate_total_maps.py | Combines lensed CIB signal from individual shells into the observed signal | None
cosmology.py | Defines cosmological parameters and functions | Not executable
utils.py | Various useful functions, mostly lensing related | Not executable
tests.py | Various tests | Not executable

## Run order ##

* 01_calculate_unlensed.py has no dependency and can be run for each desired frequency
* 02_process_kappa_maps.py must be run before 03_calculate_lensed.py. Needs to be run only once
  per set of lensing convergence maps.
* 03_calculate_lensed.py should be run for each frequency and shell
* 04_calculate_total_maps.py should be run as the very last. Currently it assumes the
  other codes were fully run for 217 GHz, 353 GHz and 545 GHz CIB signal.
