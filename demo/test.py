import os
import subprocess
import sys

import flake8.api.legacy as flake8
import pytest

convergence_demos = [
    "convergence/field_advection.py",
    "convergence/manufactured_poisson.py",
    "convergence/heat_equation_particles.py",
    "convergence/boussinesq_manufactured_solution.py",
    "convergence/compressible_boussinesq_manufactured_solution.py",
    "convergence/incompressible_thermochemical_particles_manufactured_solution.py"
]


benchmark_demos = [
    "benchmark/rayleigh_taylor.py",
    "benchmark/blankenbach_case3.py",
    "benchmark/incompressible_thermochemical.py",
    # "benchmark/compressible_rayleigh_taylor.py",
]


# Build list of demo programs
def build_demo_list(demo_dirs):
    demos = []
    for demo_dir in demo_dirs:
        if os.path.isfile(demo_dir):
            demo_dir, demo_file = os.path.split(demo_dir)
            demos.append((os.path.abspath(demo_dir), demo_file))
            continue
        demo_files = [f for f in os.listdir(demo_dir) if f.endswith(".py")]
        for demo_file in demo_files:
            demos.append((os.path.abspath(demo_dir), demo_file))

    return demos


def dispatch_demo(path, name):
    ret = subprocess.run([sys.executable, name],
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
    return ret


@pytest.mark.serial
@pytest.mark.parametrize("path,name", build_demo_list(convergence_demos))
def test_convergence_demos(path, name):
    ret = dispatch_demo(path, name)
    assert ret.returncode == 0


@pytest.mark.serial
@pytest.mark.parametrize("path,name", build_demo_list(benchmark_demos))
def test_benchmark_demos(path, name):
    ret = dispatch_demo(path, name)
    assert ret.returncode == 0


@pytest.mark.serial
@pytest.mark.parametrize("path,name", build_demo_list(convergence_demos))
def test_flake8_demos(path, name):
    report = flake8.get_style_guide().check_files(
        paths=[os.path.join(path, name)])
    assert len(report.get_statistics("F")) == 0
    assert len(report.get_statistics("E")) == 0
