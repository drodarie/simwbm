import imp
from setuptools import find_packages, setup

VERSION = imp.load_source("", "simwbm/version.py").__version__

setup(name='simwbm',
      version=VERSION,
      url='https://bluebrain.epfl.ch/',
      author='Dimitri RODARIE',
      author_email='dimitri.rodarie@epfl.ch',
      description='Simulation of the Point neuron model of the whole mouse brain',
      install_requires=['braindb>=1.0.0',
                        'h5py>=3.6.0',
                        'mpi4py>=3.1.3',
                        'numpy>=1.22.1',
                        'neuron>=8.0',
                        'nest<=2.20.2'
                        'matplotlib>=3.5.1'],
      packages=find_packages())
