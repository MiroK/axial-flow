from dolfin import *


class ALEMap(object):
    '''Harmonic extension of given bdry data'''
    def __init__(self, V, bdries, bcs, weight=True):
        mesh = V.mesh()

        if weight:
            r = SpatialCoordinate(mesh)[0]
        else:
            r = Constant(1)

        u, v = TrialFunction(V), TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        L = inner(Constant((0, 0)), v)*dx
        # Update bcs
        bc_tags = [b[0] for b in bcs]
        self.bc_expr = [b[1] for b in bcs]

        bcs = [DirichletBC(V, value, bdries, tag) for tag, value in zip(bc_tags, self.bc_expr)]
        assembler = SystemAssembler(a, L, bcs)

        A = PETScMatrix()
        b = PETScVector()
        assembler.assemble(A)

        solver = PETScKrylovSolver('cg', 'amg')
        solver.set_operators(A, A)
        self.solver = solver

        self.b = b
        self.assembler = assembler

    def compute(self, f, t):
        for expr in self.bc_expr: expr.t = t

        self.assembler.assemble(self.b)
        self.solver.solve(f.vector(), self.b)
        return x

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    import sympy as sp

    # Reference coordinates in tems of which bcs are defined
    R, Z, t = sp.symbols('x[0], x[1], t')
    # Parameters of the mapping
    A, B, x0, v = sp.symbols('A, B, x0, v')
    # Displacement, 2 shifted bumps
    u_R = A*sp.exp(-(Z-v*t)**2/B) + A*sp.exp(-(Z-v*t-0.5)**2/B)
    u_Z = sp.S(0)
    # Velocity 
    v_R = u_R.diff(t, 1)
    v_Z = u_Z.diff(t, 1)

    A_value = 0.025
    B_value = 0.01
    v_value = 0.8
    x0_value = 0.25
    # Displacement and velocity expressions
    u = Expression((sp.printing.ccode(u_R), sp.printing.ccode(u_Z)), 
                   degree=1,
                   A=A_value, B=B_value, x0=x0_value, v=v_value, t=0.)

    v = Expression((sp.printing.ccode(v_R), sp.printing.ccode(v_Z)), 
                   degree=1,
                   A=A_value, B=B_value, x0=x0_value, v=v_value, t=0.)

    mesh = RectangleMesh(Point(0.2, 0), Point(0.5, 1), 10, 20)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    bdries = FacetFunction('size_t', mesh, 0)
    DomainBoundary().mark(bdries, 1)
    CompiledSubDomain('near(x[0], 0.2)').mark(bdries, 2)

    ale = ALEMap(V, bdries, [(1, Constant((0, 0))), (2, u)])

    mesh0 = Mesh(mesh)
    pp = plot(mesh0)
    T = 1/v_value       # Time it takes to cover the domain

    for k in range(2):
        for t in np.linspace(0, T, 1000):
            x = Function(V)
            ale.compute(x, t)

            mesh0 = Mesh(mesh)
            ALE.move(mesh0, x)

            pp.plot(mesh0)
    interactive()
