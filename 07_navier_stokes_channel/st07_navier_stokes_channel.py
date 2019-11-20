from pathlib import Path

import numpy as np

import skfem
from skfem.models.poisson import vector_laplace, laplace
from skfem.models.general import divergence

import meshio


@skfem.bilinear_form
def vector_mass(u, du, v, dv, w):
    return sum(v * u)


@skfem.bilinear_form
def port_pressure(u, du, v, dv, w):
    return sum(v * (u * w.n))


p_inlet = 8.

mesh = skfem.MeshTri()
mesh.refine(3)

boundary = {
    'inlet':  mesh.facets_satisfying(lambda x: x[0] == 0),
    'outlet': mesh.facets_satisfying(lambda x: x[0] == 1),
    'wall':  mesh.facets_satisfying(lambda x: np.logical_or(x[1] == 0,
                                                            x[1] == 1)),
}
boundary['ports'] = np.concatenate([boundary['inlet'],
                                    boundary['outlet']])

element = {'u': skfem.ElementVectorH1(skfem.ElementTriP2()),
           'p': skfem.ElementTriP1()}
basis = {**{v: skfem.InteriorBasis(mesh, e, intorder=4)
            for v, e in element.items()},
         **{label: skfem.FacetBasis(mesh, element['u'],
                                    facets=boundary[label])
            for label in ['inlet', 'outlet']}}


M = skfem.asm(vector_mass, basis['u'])
L = {'u': skfem.asm(vector_laplace, basis['u']),
     'p': skfem.asm(laplace, basis['p'])}
B = -skfem.asm(divergence, basis['u'], basis['p'])
P = B.T + skfem.asm(port_pressure,
                    *(skfem.FacetBasis(mesh, element[v],
                                       facets=boundary['ports'], intorder=3)
                      for v in ['p', 'u']))

t_final = 1.
dt = .05

dirichlet = {'u': basis['u'].get_dofs(boundary['wall']).all(),
             'p': np.concatenate([
                 basis['p'].get_dofs(boundary['ports']).all()])}
inlet_pressure_dofs = basis['p'].get_dofs(boundary['inlet']).all()

uv_, p_ = (np.zeros(basis[v].N) for v in element.keys())  # penultimate
p__ = np.zeros_like(p_)  # antepenultimate

K = M / dt + L['u']

t = 0

with meshio.XdmfTimeSeriesWriter(
        Path(__file__).with_suffix('.xdmf').name) as writer:

    writer.write_points_cells(mesh.p.T, {'triangle': mesh.t.T})

    while t < t_final:

        t += dt

        # Step 1: Momentum prediction (Ern & Guermond 2002, eq. 7.40, p. 274)

        uv = skfem.solve(*skfem.condense(
            K, (M / dt) @ uv_ - P @ (2 * p_ - p__),
            np.zeros_like(uv_), D=dirichlet['u']))

        # Step 2: Projection (Ern & Guermond 2002, eq. 7.41, p. 274)

        dp = np.zeros(basis['p'].N)
        dp[inlet_pressure_dofs] = p_inlet - p_[inlet_pressure_dofs]

        dp = skfem.solve(*skfem.condense(L['p'], B @ uv,
                                         dp, D=dirichlet['p']))

        # Step 3: Recover pressure and velocity (E. & G. 2002, p. 274)

        p = p_ + dp
        print(min(p), '<= p <= ', max(p))

        du = skfem.solve(*skfem.condense(M / dt, -P @ dp, D=dirichlet['u']))
        u = uv + du

        uv_ = uv
        p_, p__ = p, p_

        # postprocessing
        writer.write_data(
            t,
            point_data={
                'pressure': p,
                'velocity': np.pad(u[basis['u'].nodal_dofs].T,
                                   ((0, 0), (0, 1)), 'constant')})

        print(min(u[::2]), '<= u <= ', max(u[::2]),
              '||v|| = ', np.linalg.norm(u[1::2]))


# References

#  Ern, A., Guermond, J.-L. (2002). _Éléments finis : théorie,
#  applications, mise en œuvre_ (Vol. 36). Paris: Springer. ISBN:
#  3540426159
