from setuptools import setup

setup(
    name='geopart',
    version='2020.1.0',
    package_dir={"geopart": "geopart"},
    packages=['geopart',
              'geopart.stokes',
              'geopart.composition',
              'geopart.energy',
              'geopart.ala'],
    url='',
    license='',
    author='Nate Sime',
    author_email='nsime@carnegiescience.edu',
    description='Utility classes and functions for geodynamics modelling '
                'with LEoPart'
)
