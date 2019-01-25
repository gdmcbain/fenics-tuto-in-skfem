Translation into scikit-fem from FEniCS of
[ft03_heat.py](https://fenicsproject.org/pub/tutorial/python/vol1/ft03_heat.py)
as discussed in [‘The heat
equation/Test problem 1: a known analytical solution ’](https://fenicsproject.org/pub/tutorial/html/._ftut1006.html#___sec49).

This is much as in [‘Heat equation’](https://kinnala.github.io/scikit-fem-docs/examples/ex19.html) in the scikit-fem manual, but:
* has evolving Dirichlet data
* has volumetric heating
* uses backward Euler rather than Crank–Nicolson
* doesn't factorize the evolution matrix
