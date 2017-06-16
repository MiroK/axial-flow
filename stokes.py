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

    a00 = mu*inner(grad(u), grad(v))*r*dx + mu*(inner(u[0], v[0])/r)*dx(degree=5)
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
    R0 = 0.2
    R1 = 0.5
    H = 1   # Hardcoded
    mu = Constant(2)

    for n in (2, 4, 8, 16, 32, 64):
        mesh = RectangleMesh(Point(R0, 0), Point(R1, H), n, 2*n)
        boundaries = FacetFunction('size_t', mesh)
        # Velocity inner
        CompiledSubDomain('near(x[0], R0)', R0=R0).mark(boundaries, 1)
        CompiledSubDomain('near(x[0], R0)', R0=R1).mark(boundaries, 2)
        CompiledSubDomain('near(x[1], H)', H=0).mark(boundaries, 3)
        CompiledSubDomain('near(x[1], H)', H=H).mark(boundaries, 4)

        bcs = {'domains': boundaries,                                       # v hardcoded
               'values':  (Constant((0, 0)), Constant((0, 0)), Constant(0), Constant(-10))}
        
        uh, ph = stokes_iter(mesh, bcs, mu)

        r = SpatialCoordinate(mesh)[0]
        print map(sqrt, map(assemble, (inner(uh[0], uh[0])*r*dx, inner(uh[1], uh[1])*r*dx)))

    # plot(uh, title='uh')
    # FIXME: MMS test
    # FIXME: NAVIER-STOKES
    # FIXME: NAVIER-STOKES with ALE
    # FIXEM: combine with surface trackit
    n
