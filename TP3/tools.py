import skfem  # for Finite Element Method
import numpy as np
import matplotlib.pyplot as plt
import random

from skfem import MeshLine, Basis, asm, enforce,solve
from skfem.element import ElementLineP1
from skfem.helpers import dot, grad
from skfem.assembly import BilinearForm, LinearForm


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


""" POD """
### USE AND ADAPT TP1 FUNCTION...

def Construct_RB(m,NumberOfSnapshots=50,NumberOfModes=30, seed = np.random.seed(42)):
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
    C = Snapshots.T@L2@Snapshots        #Q: a quel point est on sûr de la formule ?

    if(C.shape != (NumberOfSnapshots, NumberOfSnapshots)) :
        print("La matrice de corrélation n'est pas de la bonne taaaaaaaille. Recommences.\n")
        exit(-1)

    # Then, we compute the eigenvalues/eigenvectors of C (EigenVectors=alpha)
    EigenValues, EigenVectors = np.linalg.eigh(C, UPLO="L") #SVD: C eigenVectors=eigenValues eigenVectors
    ## Vecteur propre stocké en colonne

    idx = EigenValues.argsort()[::-1] # sort the eigenvalues
    print(idx,"idx \n")
    TotEigenValues = EigenValues[idx] # Valeurs propres réordonnées
    TotEigenVectors = EigenVectors[:, idx] # Réordonnement selon les valeurs propres

    # retrieve N=NumberOfModes first eigenvalues
    EigenValues = np.array([TotEigenValues[i] for i in range(NumberOfModes)]) # On prend les NumberOfModes première valeurs
    EigenVectors = np.array([TotEigenVectors[:,i] for i in range(NumberOfModes)]) # Les vecteurs propres associés

    print("eigenvalues: ",EigenValues)

    RIC = 1 - sum(EigenValues[i] for i in range(NumberOfModes))/sum(lbd for lbd in TotEigenValues) #must be close to 0
    print("Relativ Information Content (must be close to 0): ",RIC)

    ChangeOfBasisMatrix = np.zeros((NumberOfSnapshots, NumberOfModes))

    for j in range(NumberOfModes):
        ChangeOfBasisMatrix[:,j] = EigenVectors[j,:]/np.sqrt(EigenValues[j]) #/ normalization
    
    ReducedBasis = Snapshots@ChangeOfBasisMatrix 

    #Id = ReducedBasis.T @ L2 @ ReducedBasis
    #print(np.allclose(Id, np.eye(NumberOfModes)))
    
    return ReducedBasis
