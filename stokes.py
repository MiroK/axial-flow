#       4
#  |----------|
#  |          |2
# 1|          |z
#  |          |
#  |----------|
#  R0   3     R1
#
# Stokes flow with axially symmetric domain above 
# No slip on 1, 2, the rest has -p*n + mu*grad(u).n = h bcs
from dolfin import *
from block import block_assemble, block_bc, block_mat
from block.iterative import MinRes
from block.algebraic.petsc import AMG, LumpedInvDiag


def stokes(mesh, bcs, mu):
    # We assume that first coordinate is r, second is z. Then gradinet in cylindrical
    # coordinates assuming that the field has no angular component is
    # [grad(u) 0;
    #  0       u_r/r] where the first block is for r, z and the last is angular. Here grad
    #  is understood as d u_i/d coord_j
    # The extra term r is from jacobian.
    assert mesh.geometry().dim() == 2
    assert len(bcs['values']) == 4

    bdries = bcs['domains']
    uW1, uW2, p3, p4 = bcs['values']

    V = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    Q = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    TH = MixedElement([V, Q])
    W = FunctionSpace(mesh, TH)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    r = SpatialCoordinate(mesh)[0]
    n = FacetNormal(mesh)

    bcs = [DirichletBC(W.sub(0), uW, bdries, tag) for tag, uW in enumerate((uW1, uW2), 1)]

    ds = Measure('ds', domain=mesh, subdomain_data=bdries)

    a = mu*inner(grad(u), grad(v))*r*dx + mu*(inner(u[0], v[0])/r)*dx
    a+= -inner(p, div(v))*r*dx - inner(p, v[0])*dx
    a+= -inner(q, div(u))*r*dx - inner(q, u[0])*dx

    f = Constant((0, 0))
    L = inner(p3, dot(v, n))*r*ds(3) + inner(p4, dot(v, n))*r*ds(4) + inner(f, v)*r*dx
    L += inner(Constant(0), q)*dx

    wh = Function(W)
    solve(a == L, wh, bcs)

    uh, ph = wh.split(deepcopy=True)

    return uh, ph


def stokes_iter(mesh, bcs, mu):
    # We assume that first coordinate is r, second is z. Then gradinet in cylindrical
    # coordinates assuming that the field has no angular component is
    # [grad(u) 0;
    #  0       u_r/r] where the first block is for r, z and the last is angular. Here grad
    #  is understood as d u_i/d coord_j
    # The extra term r is from jacobian.
    assert mesh.geometry().dim() == 2
    assert len(bcs['values']) == 4

    bdries = bcs['domains']
    uW1, uW2, p3, p4 = bcs['values']

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    r = SpatialCoordinate(mesh)[0]
    n = FacetNormal(mesh)

    bcs = [DirichletBC(V, uW, bdries, tag) for tag, uW in enumerate((uW1, uW2), 1)]

    ds = Measure('ds', domain=mesh, subdomain_data=bdries)

    a00 = mu*inner(grad(u), grad(v))*r*dx + mu*(inner(u[0], v[0])/r)*dx
    a01 = -inner(p, div(v))*r*dx - inner(p, v[0])*dx
    a10 = -inner(q, div(u))*r*dx - inner(q, u[0])*dx

    f = Constant((0, 0))
    L0 = inner(p3, dot(v, n))*r*ds(3) + inner(p4, dot(v, n))*r*ds(4) + inner(f, v)*r*dx
    L1 = inner(Constant(0), q)*dx

    AA = block_assemble([[a00, a01], [a10, 0]])
    bb = block_assemble([L0, L1])

    block_bc([bcs, []], True).apply(AA).apply(bb)

    # ------------------------------------------------------------------------------------

    a = mu*inner(grad(u), grad(v))*r*dx + mu*(inner(u[0], v[0])/r)*dx
    L = inner(f, v)*r*dx
    A, _ = assemble_system(a, L, bcs)

    m = inner(p, q)*r*dx
    M = assemble(m)

    Minv = LumpedInvDiag(M)

    BB = block_mat([[AMG(A), 0], [0, Minv]])

    AAinv = MinRes(AA, precond=BB, tolerance=1e-10, maxiter=200, show=2)

    # Compute solution
    U, P = AAinv * bb

    uh = Function(V, U)
    ph = Function(Q, P)

    return uh, ph

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    R0 = 0.2
    R1 = 0.5
    H = 1   # Hardcoded
    mu = Constant(2)

    uh0, ph0 = None, None
    for n in (2, 4, 8, 16, 32, 64):
        mesh = RectangleMesh(Point(R0, 0), Point(R1, H), n, 2*n)
        boundaries = FacetFunction('size_t', mesh)
        # Velocity inner
        CompiledSubDomain('near(x[0], R0)', R0=R0).mark(boundaries, 1)
        CompiledSubDomain('near(x[0], R0)', R0=R1).mark(boundaries, 2)
        CompiledSubDomain('near(x[1], H)', H=0).mark(boundaries, 3)
        CompiledSubDomain('near(x[1], H)', H=H).mark(boundaries, 4)

        A_ = 10/4/mu(0)
        B_ = -A_*(R1**2-R0**2)/np.log(R1/R0)

        bcs = {'domains': boundaries,                                       # v hardcoded
               'values':  (Constant((0, 0)),
                           Constant((0, 0)),
                           Constant(0),
                           Constant(-10))
               }
        
        uh, ph = stokes_iter(mesh, bcs, mu)

        if uh0 is not None:
            V, Q = uh.function_space(), ph.function_space()

            mesh = V.mesh()
            r = SpatialCoordinate(mesh)[0]

            e = uh - interpolate(uh0, V)
            eu = sqrt(assemble(inner(grad(e), grad(e))*r*dx))
            eu /= sqrt(assemble(inner(grad(uh), grad(uh))*r*dx))

            e = ph - interpolate(ph0, Q)
            ep = sqrt(assemble(inner(e, e)*r*dx))
            ep /= sqrt(assemble(inner(ph, ph)*r*dx))

            print 'Difference', eu, ep
            print 'L2 pressure', errornorm(Expression('10*x[1]', degree=5), ph, 'L2')
            print 'L2 ur', sqrt(assemble(inner(uh[0], uh[0])*dx))   # 0
            # NOTE: So now the question is what is uz converging to, ignore temporarily
        uh0, ph0 = uh, ph

    # plot(uh, title='uh')
    # interactive()

    # OKAY - independence on z
    # import matplotlib.pyplot as plt
    # r = np.linspace(R0, R1, 100)
    # plt.figure()
    # for z in (0., 0.2, 0.4, 0.6, 0.8, 1.0):
    #     plt.plot(r, [uh(ri, z)[1] for ri in r])
    # plt.show()

    # FIXME: NAVIER-STOKES
    # FIXME: NAVIER-STOKES with ALE
    # FIXME: combine with surface trackit

