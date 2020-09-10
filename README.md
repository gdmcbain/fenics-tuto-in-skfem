reproducing the [examples](https://fenicsproject.org/pub/tutorial/python/vol1) from [_Solving PDEs in Python—The FEniCS Tutorial Volume 1_](https://fenicsproject.org/pub/tutorial/html/ftut1.html) tutorial in [scikit-fem](https://github.com/kinnala/scikit-fem)

This idea was hatched in https://github.com/kinnala/scikit-fem/issues/31.

# Set-up

Using Miniconda.

```shell
conda create -n fenics-tuto-in-skfem python h5py
conda activate fenics-tuto-in-skfem
conda install scipy matplotlib
pip install scikit-fem pygmsh 
conda install sympy
pip install pyamgcl  # for 08
```
