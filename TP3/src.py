import skfem  # for Finite Element Method
import numpy as np
import matplotlib.pyplot as plt
import random

"""
Elliptic diffusion problem (1D) with scikit-fem:
    -d/dx ( a(mu) * du/dx ) = f(x)   on (0, 1)
    u(0) = u(1) = 0  (Dirichlet)

We solve it for two parameters mu1, mu2 (via a(mu)),
and also plot an intermediate solution between them.
"""
from skfem import MeshLine, Basis, asm, enforce,solve
from skfem.element import ElementLineP1
from skfem.helpers import dot, grad
from skfem.assembly import BilinearForm, LinearForm

# -----------------------
# Problem setup
# -----------------------
Ne = 14 # number of mesh points 

m = MeshLine(np.linspace(0.0, 0.5, Ne + 1))
basis = Basis(m, ElementLineP1())

# Dirichlet boundary DOFs 
D = basis.get_dofs().all()  # all dofs

# Parameter-dependent diffusion coefficient
def A_mu(mu):
    return 0.1 + mu

# Sinusoidal source term : compute int f*v
@LinearForm
def rhs(v, w):
    x = w.x[0]
    f = np.sin(2*np.pi*x) # sinusoidal source
    return f*v


@BilinearForm
def diffusion(u, v, w):
    return dot(grad(u), grad(v))


def FEMassembling(m):
    """
    m= mesh
    return A,b (no parameter dependance)
    """

    basis = Basis(m, ElementLineP1())
    
    A = asm(diffusion, basis) #assembling stiffness
    b = asm(rhs, basis)
    return A,b


def FEMsolve(A,b,m,mu):
    # solve Au=b
    # Apply homogeneous Dirichlet at x=0,0.5
    
    A, b = enforce(A_mu(mu)*A,b, D=m.boundary_nodes()) #enforce boundary conditions on A and b
    u = solve(A, b)
    return u


# -----------------------------------
# Solve for two mus + an intermediate
# -----------------------------------
mu1 = 0.5
mu2 = 1.5
theta = 0.2  # intermediate parameter weight 
mu_mid = 1.

A,b = FEMassembling(m)
u1 = FEMsolve(A, b, m, mu1)

A,b = FEMassembling(m)
u2 = FEMsolve(A,b,m,mu2)

A,b = FEMassembling(m)
umid = FEMsolve(A,b,m,mu_mid)

def sol1(x, mu) : 
    return np.sin(2*np.pi*x)/((mu+0.1)*(2*np.pi)**2)


#linear combination in solution space:
theta2 = 5
u_lin = theta*u1 + theta2*u2

x_fine = basis.doflocs[0] 
plt.figure(figsize=(8, 4.5))
#plt.plot(x_fine, u1, lw=2, label=fr"$u(\mu_1)$, $\mu_1={mu1}$, $A={A_mu(mu1):.3g}$")
plt.plot(x_fine, sol1(x_fine, mu1), label = 'Solution exacte')
plt.plot(x_fine, u2, lw=2, label=fr"$u(\mu_2)$, $\mu_2={mu2}$, $A={A_mu(mu2):.3g}$")
plt.plot(x_fine, umid, lw=2, label=fr"$u(\mu_{{mid}})$, $\mu_{{mid}}={mu_mid}$, $A={A_mu(mu_mid):.3g}$")
plt.plot(x_fine, u_lin, "--", lw=2, label=fr"linear blend $\alpha_1 u(\mu_1)+\alpha_2\ u(\mu_2)$")


plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(r"Elliptic diffusion: $-(A(\mu)u')' = \sin(2\pi x)$,  $u(0)=u(0.5)=0$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

""" POD """
### USE AND ADAPT TP1 FUNCTION...

def Construct_RB(NumberOfSnapshots=50,NumberOfModes=30,m=m, seed = np.random.seed(42)):
    """
    NumberOfSnapshots= Training set
    NumberOfModes= N 
    m=mesh
    """
    #print("number of modes: ",NumberOfModes)
    basis = Basis(m, ElementLineP1())
    A,b = FEMassembling(m)
    
    Snapshots=np.zeros((m.nvertices, NumberOfSnapshots))
    for i in range(NumberOfSnapshots):
        mu = 10*(np.random.rand() + 1) #random coefficient in [1,10] 
        U = FEMsolve(A,b,m,mu)
        Snapshots[:,i] = U
        
    # print("last parameter:",mu)

    ## SVD ##

    #(u,v)_L2=v^T M u
    @BilinearForm
    def massVelocity(u, v, _):
        return u*v
    
    L2=massVelocity.assemble(basis)

    # We first compute the correlation matrix C_ij = (u_i,u_j)
    C = Snapshots.T@L2@Snapshots
    if(C.shape != (NumberOfSnapshots, NumberOfSnapshots)) :
        print("La matrice de corrélation n'est pas de la bonne taaaaaaaille. Recommences.\n")
        exit(-1)

    # Then, we compute the eigenvalues/eigenvectors of C (EigenVectors=alpha)
    EigenValues, EigenVectors = np.linalg.eigh(C, UPLO="L") #SVD: C eigenVectors=eigenValues eigenVectors
    ## Vecteur propre stocké en colonne

    idx = EigenValues.argsort()[::-1] # sort the eigenvalues
    TotEigenValues = EigenValues[idx] # Valeurs propres réordonnées
    TotEigenVectors = EigenVectors[:, idx] # Réordonnement selon les valeurs propres

    # retrieve N=NumberOfModes first eigenvalues
    EigenValues = np.array([TotEigenValues[i] for i in range(NumberOfModes)]) # On prend les NumberOfModes première valeurs
    EigenVectors = np.array([TotEigenVectors[:,i] for i in range(NumberOfModes)]) # Les vecteurs propres associés

    #print("eigenvalues: ",EigenValues)

    RIC = 1 - sum(EigenValues[i] for i in range(NumberOfModes))/sum(lbd for lbd in TotEigenValues) #must be close to 0
    #print("Relativ Information Content (must be close to 0): ",RIC)

    ChangeOfBasisMatrix = np.zeros((NumberOfSnapshots, NumberOfModes))

    for j in range(NumberOfModes):
        ChangeOfBasisMatrix[:,j] = EigenVectors[j,:]/np.sqrt(EigenValues[j]) #/ normalization
    
    ReducedBasis = Snapshots@ChangeOfBasisMatrix 

    #Id = ReducedBasis.T @ L2 @ ReducedBasis
    #print(np.allclose(Id, np.eye(NumberOfModes)))
    
    return ReducedBasis

## test POD

m = MeshLine(np.linspace(0.0, 0.5, Ne + 1))
basis = Basis(m, ElementLineP1())
ReducedBasis= Construct_RB(NumberOfModes=2, m=m, seed = np.random.seed(42))
x_fine = basis.doflocs[0] 

plt.figure(figsize=(8, 4.5))
plt.plot(x_fine, ReducedBasis[:,0], lw=2, label=fr"$\Phi_1$")
plt.plot(x_fine, ReducedBasis[:,1], lw=2, label=fr"$\Phi_2$")


plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(r"Elliptic diffusion: $-(A(\mu)u')' = \sin(2\pi x)$,  $u(0)=u(0.5)=0$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

def solve_fem_rom(A,b,mu, Phi,m):
    Q_mu = A_mu(mu)*Phi.T@A@Phi
    b_mu = Phi.T@b

    #u_rom = np.linalg.lstsq(Q_mu, b_mu)
    u_rom = np.linalg.inv(Q_mu)@b_mu
    return u_rom #( or return only u_rom)

mu = mu_mid
m = MeshLine(np.linspace(0.0, 0.5, Ne + 1))
basis = Basis(m, ElementLineP1())
A,b=FEMassembling(m)
Phi = Construct_RB(NumberOfSnapshots=5, NumberOfModes=3,m=m)
u_rom=solve_fem_rom(A,b,mu_mid, Phi,m)
u_proj = Phi @ u_rom
x_fine = basis.doflocs[0]

plt.plot(x_fine,u_proj, lw=2, label="$urom_{mid}$")
#plt.plot(x_fine, umid, lw=2, label=fr"$u(\mu_{{mid}})$, $\mu_{{mid}}={mu_mid}$, $A={A_mu(mu_mid):.3g}$")
plt.plot(x_fine, u_lin, "--", lw=2, label=fr"linear blend $\alpha_1\ u(\mu_1)+\alpha_2\ u(\mu_2)$")


plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(r"Elliptic diffusion: $-(A(\mu)u')' = \sin(2\pi x)$,  $u(0)=u(0.5)=0$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#### Convergence

mu = 0.6
@BilinearForm
def massVelocity(u, v, _):
    return u*v

err_true=[]
err_rom=[]
# choose grid sizes to test
Ns = [5,10,30,60,90,300,900] # adapt as you want
for n in Ns:
    print("n",n)
    m = MeshLine(np.linspace(0.0, 0.5, n + 1))
    basis = basis = Basis(m, ElementLineP1())
    L2=massVelocity.assemble(basis)
    xc = basis.doflocs[0] 
    
    A, b = FEMassembling(m)
    U = FEMsolve(A,b,m,mu)
    
    Phi = Construct_RB(NumberOfSnapshots=5, NumberOfModes=3,m=m)
    U_rom = solve_fem_rom(A,b,mu,Phi,m)
    Uproj = Phi@U_rom

    points_new = xc
    u_exact= sol1(points_new,mu) #compute true solution
   
    ## print error
    true_error = np.abs(u_exact - U)
    l2_true_error=np.sqrt(true_error@L2.dot(true_error))
    print(l2_true_error)
    rom_error = np.abs(u_exact - Uproj)
    l2_rom_error=np.sqrt(rom_error@L2.dot(rom_error))
    print(l2_rom_error)


   # L2 errors (cellwise)
    err_true.append(l2_true_error)
    err_rom.append(l2_rom_error)

# ---------------------------
# Plot log-log convergence
# ---------------------------
hs = np.array(Ns)
err_true = np.array(err_true)
err_rom = np.array(err_rom)
coefs = np.polyfit(np.log(hs), np.log(err_rom),1)
print(coefs[0])

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
