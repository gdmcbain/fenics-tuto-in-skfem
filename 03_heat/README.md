Translation into scikit-fem from FEniCS of
[ft03_heat.py](https://fenicsproject.org/pub/tutorial/python/vol1/ft03_heat.py)
as discussed in [‘The heat
equation’](https://fenicsproject.org/pub/tutorial/html/._ftut1006.html#ch:fundamentals:diffusion).

This is much as in [‘Heat equation’](https://kinnala.github.io/scikit-fem-docs/examples/ex19.html) in the scikit-fem manual, but:
* has evolving Dirichlet data
* has volumetric heating
* uses backward Euler rather than Crank–Nicolson

For the moment, this is done without factorizing the evolution matrix
as in the scikit-fem example which requires
[scikit-sparse](https://github.com/scikit-sparse/scikit-sparse). (TODO)
