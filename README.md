# GeoPart - Geophysics & Particles

## What does it do?

GeoPart provides utility functions to complement the
[LEoPart](https://bitbucket.org/jakob_maljaars/leopart) library.

Further utility functions are provided to complement the
[LEoPart fork](https://bitbucket.org/nate-sime/leopart) library
facilitating compressible flows such as the anelastic liquid approximation.

## Features

GeoPart leverages LEoPart for:

* Pointwise divergence free finite element velocity solutions of the Stokes
system by the hybrid discontinuous Galerkin (HDG) method.
* Exactly mass conserving projection methods for advection of fields.
* Solution of block local/global finite element (FE) problems found in HDG
formulations by static condensation.

GeoPart provides:

* A collection of commonly used elements for the Stokes finite element (FE) 
system.
* Encapsulation of ℓ₂ and PDE-Constrained ℓ₂ projection of particle data by
LEoPart.
* A suite of tests and demonstrations of optimal error convergence rates
using exactly mass conserving methods in a geodynamics context.
 
# Installation

### Dependencies

GeoPart requires:

* [FEniCS](https://fenicsproject.org/)
    - [FIAT](https://github.com/FEniCS/fiat)
    - [UFL](https://bitbucket.org/fenics-project/ufl)
    - [dijitso](https://bitbucket.org/fenics-project/dijitso)
    - [FFC](https://bitbucket.org/fenics-project/ffc)
    - [DOLFIN](https://bitbucket.org/fenics-project/dolfin)
* [LEoPart fork](https://bitbucket.org/nate-sime/leopart)
* [dolfin_dg](https://bitbucket.org/nate-sime/dolfin_dg)
* [numpy](https://github.com/numpy/numpy)

## Install to existing environment

Follow the typical procedure:

```
cd geopart
python3 setup.py install --user
```

## Install with Docker

*For extensive documentation regarding using FEniCS in a Docker environment see
[here](https://fenics.readthedocs.io/projects/containers)*

### Example

Acquire the stable FEniCS docker image `quay.io/fenicsproject/stable:latest`.

Compile and install LEoPart

```
git clone https://bitbucket.org/nate-sime/leopart.git
cd leopart/source/cpp
cmake .
make
cd ../..
python3 setup.py install
```

Compile and install `dolfin_dg`

```
git clone https://bitbucket.org/nate-sime/dolfin_dg.git
cd dolfin_dg
python3 setup.py install
```

Compile and install `geopart`

```
git clone https://bitbucket.org/nate-sime/geopart.git
cd geopart
python3 setup.py install
```

# References and citing

N. Sime, J.M. Maljaars, C.R. Wilson and P.E. van Keken \
*An exactly mass conserving and pointwise divergence free velocity
method: application to compositional buoyancy driven flow problems in
geodynamics* \
[Geochemisry, Geophysics, Geosystems, 2021](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020GC009349) \
([ESSOAr](https://www.essoar.org/doi/10.1002/essoar.10503932.1))

N. Sime, C.R. Wilson and P.E. van Keken \
*A pointwise divergence free momentum method for fileds advected by tracers
using the compressible aneslatic liquid approximation* \
In preparation

Demonstrations of exact mass conservation and pointwise divergence flow are
available in `demo/convergence/`.

The Rayleigh Taylor benchmark demonstration in available in `demo/benchmark/rayleigh_taylor.py`

## Tests

[![geopart](https://circleci.com/bb/nate-sime/geopart.svg?style=shield)](https://circleci.com/bb/nate-sime/geopart)

Unit tests are available in `test/unit/`

# Licence

LGPL-3.0
