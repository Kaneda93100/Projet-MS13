# import packages

import skfem  # for Finite Element Method
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
from scipy.sparse.linalg import factorized
import time

"""
Elliptic diffusion problem (2D) with scikit-fem:

$ -\nabla \cdot ( A(x,\mu) \nabla u) = g(x)  \ \mathrm{ on }\ (0, 1)^2$ 
$    u(0) = u(1) = 0  (Dirichlet)$

"""

from skfem import MeshTri, Basis, asm, enforce,solve
from skfem.element import ElementTriP1
from skfem.helpers import dot, grad
from skfem.assembly import BilinearForm, LinearForm
from skfem import solve



# l2 scalar product in 2D
@BilinearForm
def massMatrix(u, v, _):
    return u*v


# -----------------------
# RHS: g = 1
# -----------------------
@LinearForm
def rhs(v, w):
    return v*1

# -----------------------
# Bilinear form
# -----------------------
@BilinearForm
def diffusion(u, v, w):
    return  dot(grad(u),grad(v))

mu_l = lambda x,y: (x<0.5)*( (y<0.5)*1 +(y>0.5)*2) + (x>0.5)*( (y<0.5)*3 + (y>0.5)*4 )

@BilinearForm
def diffusion_mu(u, v, w):
    # mu en fonction de x
    mu = mu_l(w.x[0],w.x[1])
    return mu*dot(grad(u),grad(v))



# -----------------------
# Assembly
# -----------------------
def FEMassembling(m,mu):
    basis = Basis(m, ElementTriP1())
    A11 = asm(diffusion_mu, basis,mu1=mu[0],mu2=mu[1])   # on Omega_11    
    b = asm(rhs, basis)                #  g = 1
    
    return A11, b, basis


# -----------------------
# Solve
# -----------------------
def FEMsolve(A11, b, basis, mu):
    
    A_bc, b_bc = enforce(A11, b, D=basis.get_dofs().all()) # Dirichlet 
    u = solve(A_bc, b_bc) #solve
    
    return u

def Construct_RB(m,mu,NumberOfSnapshots=100,NumberOfModes=20):

    #print("number of modes: ",NumberOfModes)
    basis = Basis(m, ElementTriP1())
    A,b,basis = FEMassembling(m,mu)
    
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
    # orthogonality test
    return ReducedBasis


