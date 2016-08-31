from distutils.core import setup

packages=[
	'',
	'pygix',
]
package_dir = {'':'lib'}

setup(
	name          =   "pygix",
	version       =   "0.1.1",
	description   =   "Grazing-incidence X-ray scattering package",
	author        =   "Thomas Dane",
	author_email  =   "dane@esrf.fr",
	packages      =   packages,
	package_dir   =   package_dir,
)
