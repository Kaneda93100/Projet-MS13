# import packages

import skfem  # for Finite Element Method
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random


import numpy as np
import matplotlib.pyplot as plt

from skfem import MeshTri, Basis, asm, enforce,solve
from skfem.element import ElementTriP1
from skfem.helpers import dot, grad
from skfem.assembly import BilinearForm, LinearForm
from skfem import solve

# -----------------------
# Diffusion coefficient a(x, mu)
# Omega_1 = {x < 0.5}, Omega_2 = {x >= 0.5}
# -----------------------
def A_piecewise(x, mu):
    if x[0] < 1/2 and x[1] == 0 : ## x in Omega_1
        return mu
    else :                         ## x in Omega_2
        return 1


# -----------------------
# RHS: g = 1
# -----------------------
@LinearForm
def rhs(v, w):
    return v

# -----------------------
# Bilinear form
# -----------------------
# Global a(u,v)
@BilinearForm
def diffusion(u, v, w):
    x = w.x[0]
    return dot(grad(u), grad(v))
    
@BilinearForm
def diffusion_1(u, v, w):
    # contribution on Omega_1 = {x < 0.5}
    return (w.x[0] < 0.5) * dot(grad(u), grad(v)) ## Adapted vectorized test for np.ndarray

@BilinearForm
def diffusion_2(u, v, w):
    # contribution on Omega_2 = {x >= 0.5}
    return (w.x[0] >= 0.5) * dot(grad(u), grad(v)) ## Adapted vectorized test for np.ndarray

# -----------------------
# Assembly
# -----------------------
def FEMassembling(m):
    basis = Basis(m, ElementTriP1())
    #Instead of global which depend on mu
    A1 = asm(diffusion_1, basis)   # on Omega_1
    A2 = asm(diffusion_2, basis)   # on Omega_2
    b = asm(rhs, basis)# g = 1
    return A1,A2, b, basis


# -----------------------
# Solve
# -----------------------
def FEMsolve(A1, A2, b, basis, mu):
    A = mu*A1 + A2
    A_bc, b_bc = enforce(A, b, D=(basis.mesh).boundary_nodes()) # enforce boundary condition
    u = solve(A_bc, b_bc) # solve
    return u

"""
POD
"""

def Construct_RB(m, NumberOfSnapshots=5,NumberOfModes=3, seed = random.seed(42)):
    
    print("number of modes: ",NumberOfModes)
    basis = Basis(m, ElementTriP1())

    A1,A2,b,basis = FEMassembling(m)
  
    Snapshots = np.zeros((m.nvertices, NumberOfSnapshots))
    for i in range(NumberOfSnapshots):
        mu = 10*random.random() #random coefficient in [0, 10] 
        Snapshots[:,i] = FEMsolve(A1, A2, b, basis, mu)
        x = 0
        
       
    print("last parameter:",mu)

    ## SVD ##

    # on (u,v)_L2
    @BilinearForm
    def massVelocity(u, v, _):
        return u * v
    L2=massVelocity.assemble(basis)
    
    C = Snapshots.T@Snapshots
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
    print("Relativ Information Content (must be close to 0): ",RIC)

    ChangeOfBasisMatrix = np.zeros((NumberOfSnapshots, NumberOfModes))

    for j in range(NumberOfModes):
        ChangeOfBasisMatrix[:,j] = EigenVectors[j,:]/np.sqrt(EigenValues[j]) #/ normalization
    
    ReducedBasis = Snapshots@ChangeOfBasisMatrix 

    return ReducedBasis