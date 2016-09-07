from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

packages=find_packages(exclude=['docs', 'tests'])

print '*'*80
print packages
print '*'*80

# double check classifiers on https://pypi.python.org/pypi?%3Aaction=list_classifiers
classifiers = ["Development Status :: 4 - Beta",
               "Intended Audience :: Developers",
               "Programming Language :: Python :: 2",
               "Programming Language :: Python :: 2.7",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GNU General Public License (GPL)",
               "License :: OSI Approved :: MIT License",
               "Topic :: Software Development :: Libraries :: Python Modules",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Operating System :: MacOS :: MacOS X",
               "Topic :: Scientific/Engineering :: Physics"
               ]

install_requires = ["numpy", "h5py", "fabio", "matplotlib", "scipy", "pyFAI"]
setup_requires = ["numpy", "cython", "pyFAI"]

setup(
    name='pygix',
    version='0.1.3',
    description='Reduction of 2D Grazing-incidence and fibre diffraction data',
    long_description=long_description,
    url='https://github.com/tgdane/pygix',
    author='Thomas Dane',
    author_email='thomasgdane@gmail.com',
    licence="GPL",
    classifiers=classifiers,
    packages=packages,
    # install_requires=install_requires,
    # setup_requires=setup_requires
    )





