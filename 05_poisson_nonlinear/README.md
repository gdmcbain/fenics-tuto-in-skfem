Translation into scikit-fem from FEniCS of
[ft05_poisson_nonlinear.py](https://fenicsproject.org/pub/tutorial/python/vol1/ft05_poisson_nonlinear.py)
as discussed in [‘A nonlinear Poisson equation’](https://fenicsproject.org/pub/tutorial/html/._ftut1007.html#ftut1:gallery:nonlinearpoisson).

The big new features here are:
* the use of [SymPy](https://sympy.org) to manufacture the right-hand side from an assumed solution
* the use of [scipy.optimize.root](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html#scipy.optimize.root) as the nonlinear solver

![exact.png](poisson_nonlinear/exact.png)(*a*) ![solution.png](poisson_nonlinear/solution.png)(*b*)

*Figure:—* (*a*) exact 1 + _x_ + 2 _y_ and (*b*) numerical solutions
