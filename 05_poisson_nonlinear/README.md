Translation into scikit-fem from FEniCS of
[ft05_poisson_nonlinear.py](https://fenicsproject.org/pub/tutorial/python/vol1/ft05_poisson_nonlinear.py)
as discussed in [‘A nonlinear Poisson equation’](https://fenicsproject.org/pub/tutorial/html/._ftut1007.html#ftut1:gallery:nonlinearpoisson).

The big new feature here is the use of [SymPy](https://sympy.org) to manufacture the right-hand side from an assumed solution.

![initial.png](poisson_nonlinear/initial.png)(*a*) ![solution.png](poisson_nonlinear/solution.png)(*b*)

*Figure:—* (*a*) initial condition and (*b*) exact solution, both should be equal to, 1 + _x_ + 2 _y_

*Figure:—* Initial condition, 1 + _x_ + 2 _y_
