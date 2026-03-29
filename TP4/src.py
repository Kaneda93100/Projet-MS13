# import packages

import skfem  # for Finite Element Method
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
import tools as t

import numpy as np
import matplotlib.pyplot as plt

from skfem import MeshTri, Basis, asm, enforce,solve
from skfem.element import ElementTriP1
from skfem.helpers import dot, grad
from skfem.assembly import BilinearForm, LinearForm
from skfem import solve

# -----------------------
# Problem setup
# -----------------------

Nx = 100
Ny = 100

# Maillage triangulaire du carré (0,1)^2
m = MeshTri.init_tensor(
    np.linspace(0.0, 1.0, Nx + 1),
    np.linspace(0.0, 1.0, Ny + 1)
)

basis = Basis(m, ElementTriP1())
# Dirichlet boundary DOFs 
D = basis.get_dofs().all()  # all dofs

# -----------------------
# Example
# -----------------------
mu = 7.60379792234498
A1,A2, b, basis = t.FEMassembling(m)
u = t.FEMsolve(A1, A2, b, basis, mu)


# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(6, 5))
m.plot(u, ax=ax, shading='gouraud')
ax.set_title(f"Solution FEM 2D, mu = {mu}")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()




print("-----------------------------------")
print("        Offline                    ")
print("-----------------------------------")


## TEST 
Nx = Ny = 100
m = MeshTri.init_tensor(
    np.linspace(0.0, 1.0, Nx + 1),
    np.linspace(0.0, 1.0, Ny + 1)
)
basis = Basis(m, ElementTriP1())
Phi= t.Construct_RB(m = m, NumberOfSnapshots=100,NumberOfModes=6)
ReducedBasis=Phi.T   

fig, ax = plt.subplots()

im=m.plot(ReducedBasis[0,:], ax=ax, shading='gouraud',colorbar=True)
fig, ax = plt.subplots()

m.plot(ReducedBasis[1,:], ax=ax, shading='gouraud',colorbar=True)


plt.show()

# POD-Galerkin

# assemble full system
A1,A2, b, basis = t.FEMassembling(m)

def solve_fem_rom(A1,A2,b,mu, Phi):
    A_off = Phi.T@(mu*A1 + A2)@Phi
    b_off = Phi.T@b

    a = np.linalg.solve(A_off, b_off)
    u_rom = Phi@a
    return a, u_rom

### show projection of u on VN

mu = 10
Nx=Ny = 100
m = MeshTri.init_tensor(
    np.linspace(0.0, 1.0, Nx + 1),
    np.linspace(0.0, 1.0, Ny + 1)
)

basis = Basis(m, ElementTriP1())
A1,A2,b,basis = t.FEMassembling(m)
Phi = t.Construct_RB(m)
a,u_proj= solve_fem_rom(A1,A2,b,mu,Phi)

fig, ax = plt.subplots()

m.plot(u_proj, ax=ax, shading='gouraud',colorbar=True)

## Compare with u 
u = t.FEMsolve(A1,A2,b,basis,mu)

fig, ax = plt.subplots()

m.plot(u, ax=ax, shading='gouraud',colorbar=True)

plt.show()


#### Convergence
mu = 0.6

err_true=[]
err_rom=[]

m_ref = MeshTri.init_tensor(
    np.linspace(0.0, 1.0, 250 + 1),
    np.linspace(0.0, 1.0, 250 + 1)
)

A1_ref, A2_ref, b_ref, basis_ref = t.FEMassembling(m_ref)
U_ref = t.FEMsolve(A1_ref,A2_ref,b_ref,basis_ref,mu)


# True solution = refined solution
u_ref_interp = basis_ref.interpolator(U_ref)

# for L2 norm
@BilinearForm
def massVelocity(u, v, _):
    return u * v

# choose grid sizes to test
Ns = [20, 30, 40, 50, 60]  # adapt as you want
for n in Ns:
    print("n",n)
    m = MeshTri.init_tensor(
    np.linspace(0.0, 1.0, n + 1),
    np.linspace(0.0, 1.0, n + 1)
)
    basis = Basis(m, ElementTriP1())

    A1, A2, b, basis = t.FEMassembling(m)
    U = t.FEMsolve(A1,A2,b,basis,mu)
 
    Phi = t.Construct_RB(m)

    _,Uproj= solve_fem_rom(A1,A2,b,mu,Phi)

    
    # Refined solution interpolated on mesh m
    X = basis.doflocs              # shape = (2, ndof)
    U_ref_on_mesh = u_ref_interp(X)

    ## print error
    L2=massVelocity.assemble(basis)
    
    true_error = U_ref_on_mesh-U
    l2_true_error= true_error.T@L2@true_error
    
    rom_error = (U_ref_on_mesh-Uproj)
    l2_rom_error = rom_error.T@L2@rom_error


   # L2 errors 
    err_true.append(l2_true_error)
    err_rom.append(l2_rom_error)

# ---------------------------
# Plot log-log convergence
# ---------------------------
hs = np.array(Ns)
err_true = np.array(err_true)
err_rom = np.array(err_rom)

plt.figure()
plt.loglog(hs, err_true, "o-", label=r"$\|u_{ref}-u\|_{L^2}$")
plt.loglog(hs, 1/(hs**2), "-", label=r"$h^2$")
plt.loglog(hs, err_rom, "s-", label=r"$\|u_{ref}-u_N\|_{L^2}$")
plt.gca().invert_xaxis()  # optional: smaller h to the right
plt.xlabel(r"$h$")
plt.ylabel(r"$L^2$ error")
plt.grid(True, which="both")
plt.legend()
plt.show()


from scipy.sparse.linalg import spsolve

def alpha_LB(mu):
    """
    Lower bound of a
    """
    return min(1,mu)


def get_interior_dofs(basis):
    """
    Interior DDL ( since test functions in H^1_0)
    """
    D = basis.get_dofs().all()
    I = np.setdiff1d(np.arange(basis.N), D)
    return D, I


def RB_solve_certified(Phi, A1, A2, F, basis, mu):
   
    D, I = get_interior_dofs(basis)

    # Full matrices
    A = mu*A1 + A2
    
    @BilinearForm
    def Stiffness(u, v,_):
        return dot(grad(u), grad(v))
    
    X = Stiffness.assemble(basis)  # Matrix for the norm ||.||_X

    # Interior DDL restriction
    A_I = A[I][:, I]
    X_I = X[I][:, I]
    F_I = F[I]

    # Restricted reduced basis
    Phi_I = Phi[I,:]
    ReducedBasis_I = Phi_I
    # Reduce system
    
    A_rb = (ReducedBasis_I.T@A_I)@ReducedBasis_I
    F_rb = ReducedBasis_I.T@F_I

    coeff = np.linalg.solve(A_rb, F_rb)

    # ROM reconstructed
    u_rb_I = ReducedBasis_I@coeff 
    u_rb = np.zeros(basis.N) # 0 at the boundary 
    u_rb[I] = u_rb_I

    # Interior residual
    r_I = F_I - A_I@u_rb_I

    # Dual norm of the residual : sqrt(r^T X^{-1} r)
    z =  spla.inv(X_I)
    dual_norm = np.sqrt((r_I.T@z)@r_I)

    # Estimateur certifié
    Delta_N = dual_norm/alpha_LB(mu) # 1/alpha * 

    return u_rb, coeff, dual_norm, Delta_N


def true_error_X_norm(U_fem, U_rb, A1, A2, basis):
    """
    Error with 
    X = A1 + A2.
    """
    D, I = get_interior_dofs(basis)
    X = A1 + A2
    X_I = X[I][:, I]

    e_I = U_fem[I] - U_rb[I]
    err_X = e_I.T@X_I@e_I
    return np.sqrt(err_X)


mu = 0.6
## test with N=2 and 3 reduced basis functions

err_true_L2 = []
err_rom_L2 = []

err_true_X = []
estimator = []
effectivity = []

hs = []

# -------------------------------------------------
# Solution de référence sur maillage fin
# -------------------------------------------------
Nref = 300
m_ref = MeshTri.init_tensor(
    np.linspace(0.0, 1.0, Nref + 1),
    np.linspace(0.0, 1.0, Nref + 1)
)

A1_ref, A2_ref, F_ref, basis_ref = t.FEMassembling(m_ref)
U_ref = t.FEMsolve(A1_ref, A2_ref, F_ref, basis_ref, mu)
u_ref_interp = basis_ref.interpolator(U_ref)
    
@BilinearForm
def StiffnessMat(u, v, _):
    return dot(grad(u), grad(v))

M=StiffnessMat.assemble(basis_ref)

# interpolateur de la solution grossière vers le maillage fin
Xref = basis_ref.doflocs
x = Xref[0]
y = Xref[1]

tol = 1e-12
interior = (
        (x > tol) & (x < 1.0 - tol) &
        (y > tol) & (y < 1.0 - tol)
)

# -------------------------------------------------
# Boucle de convergence
# -------------------------------------------------
Ns = [20, 40, 60, 80]

for n in Ns:
    print("n =", n)
    hs.append(n)

    m = MeshTri.init_tensor(
    np.linspace(0.0, 1.0, n + 1),
    np.linspace(0.0, 1.0, n + 1)
)

    # reduced basis
    Phi = t.Construct_RB(m)
    # FEM + ROM + certification
    
    A1, A2, F, basis = t.FEMassembling(m)
    U_h = t.FEMsolve(A1, A2, F, basis, mu)
    
    # ROM + certification
    U_rb, coeff, dual_norm, Delta_N = RB_solve_certified(Phi, A1, A2, F, basis, mu)

    #Interpolated FEM on refined mesh
    Xdof = basis.doflocs

    u_h_interp = basis.interpolator(U_h)
    U_on_ref = np.zeros(basis_ref.N)
    U_on_ref[interior] = u_h_interp(Xref[:, interior])

    u_rb_interp = basis.interpolator(U_rb)
    U_rb_on_ref = np.zeros(basis_ref.N)
    U_rb_on_ref[interior] = u_rb_interp(Xref[:, interior])

    L2 = massVelocity.assemble(basis_ref)
    
    e_true_ref = U_ref - U_on_ref
    true_error = e_true_ref.T@L2@e_true_ref
   
    e_rom_ref = U_rb_on_ref - U_ref
    rom_error = e_rom_ref.T@L2@e_rom_ref


    err_true_L2.append(true_error) ## 
    err_rom_L2.append(rom_error)

    # -------------------------
    # True error in X norm
    # -------------------------
    err_X = true_error_X_norm(U_h, U_rb, A1, A2, basis)

    
    err_true_X.append(err_X)
    estimator.append(Delta_N)

    eff = Delta_N / err_X if err_X > 1e-14 else np.nan
    effectivity.append(eff)

    print(f"  ||u_ref - u_h||_L2 = {l2_true_error:.3e}")
    print(f"  ||u_ref - u_N||_L2 = {l2_rom_error:.3e}")
    print(f"  ||u_h - u_N||_X    = {err_X:.3e}")
    print(f"  ||r_N||_X'         = {dual_norm:.3e}")
    print(f"  Delta_N(mu)        = {Delta_N:.3e}")
    print(f"  effectivity        = {eff:.3e}")



hs = np.array(hs)

err_true_L2 = np.array(err_true_L2)
err_rom_L2 = np.array(err_rom_L2)

plt.figure(figsize=(7, 5))
plt.loglog(hs, err_true_L2, "o-", label=r"$\|u_{ref}-u_h\|_{L^2}$")
plt.loglog(hs, 1/(hs**2), "--", label=r"$h^2$")
plt.loglog(hs, err_rom_L2, "s-", label=r"$\|u_{ref}-u_N\|_{L^2}$")

plt.gca().invert_xaxis()
plt.xlabel(r"$h$")
plt.ylabel(r"Error $L^2$")
plt.grid(True, which="both")
plt.legend()
plt.title(" FEM / ROM in $L^2$ norm")
plt.show()



err_true_X = np.array(err_true_X)
estimator = np.array(estimator)
effectivity = np.array(effectivity)

plt.figure(figsize=(7, 5))
plt.loglog(hs, err_true_X, "o-", label=r"True error $\|u_h-u_N\|_X$")
plt.loglog(hs, estimator, "s--", label=r"Estimator $\Delta_N(\mu)$")

plt.gca().invert_xaxis()
plt.xlabel(r"$h$")
plt.ylabel(r"estimator")
plt.grid(True, which="both")
plt.legend()
plt.title("Certification a posteriori ")
plt.show()

plt.figure(figsize=(7, 4))
plt.semilogx(hs, effectivity, "o-")
plt.gca().invert_xaxis()
plt.xlabel(r"$h$")
plt.ylabel("Effectivity")
plt.grid(True, which="both")
plt.title(r"Effectivity $\Delta_N / \|u_h-u_N\|_X$")
plt.show()
