# import packages

import numpy as np
import matplotlib as plt
import tools as VF

#Draw one solution   
mu = (0.99, 0.8, 0.2, 0.78)
xc, yc,M,b = VF.assemble_tpfa(Nx = 30, Ny = 25, mu = mu)
U = VF.solve_tpfa(M,b,xc,yc)

plt.figure()
plt.title("TPFA solution u_K (Dirichlet u=0 on ∂Ω)")
plt.imshow(U.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
plt.colorbar()
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.show()


# ============================================================
# POD
# ============================================================

def Construct_RB(NumberOfSnapshots=50,Nx=50,Ny=50,NumberOfModes=10):
    #Training set Ntrain = NumberOfSnapshots
    print("number of modes: ",NumberOfModes) # N
    mu = (0.99, 0.8, 0.2, 0.78) #first parameter

    Snapshots=[]
    #---------------------------------#
    #      Generate the snapshots     #
    #---------------------------------#
    for i in range(NumberOfSnapshots):
        _, _,M,b = solve_tpfa()
        U = ...
        Snapshots.append(U.flatten(order="F"))
        mu = [...] #random coefficients in [0, 1] 


    #---------------------------------#
    #      POD                        #
    #---------------------------------#

    #(u,v)_L2=sum_K|K| u_k v_k 
    volK = ... #|K|

    #  snapshot correlation matrix C_ij = (u_i,u_j)
    CorrelationMatrix = np.zeros((NumberOfSnapshots, NumberOfSnapshots))
    for i, snapshot1 in enumerate(Snapshots):
        for j, snapshot2 in enumerate(Snapshots):
            if i >= j:
                CorrelationMatrix[i, j] = ...
                CorrelationMatrix[j, i] = ...


    # Then, we compute the eigenvalues/eigenvectors of C (EigenVectors=alpha)
    EigenValues, EigenVectors = np.linalg.eigh(CorrelationMatrix, UPLO="L") #SVD: C eigenVectors=eigenValues eigenVectors

    idx = EigenValues.argsort()[::-1] # sort the eigenvalues
    TotEigenValues = EigenValues[idx]
    TotEigenVectors = EigenVectors[:, idx]

    # retrieve N=NumberOfModes first eigenvalues
    EigenValues = ...
    EigenVectors = ...

    #print("eigenvalues: ",EigenValues)

    RIC = ... #must be close to 0
    print("Relativ Information Content (must be close to 0): ",RIC)

    ChangeOfBasisMatrix = np.zeros((NumberOfModes,NumberOfSnapshots))

    for j in range(NumberOfModes):
        ChangeOfBasisMatrix[j,:] = EigenVectors[:,j]/... #/ normalization

    ReducedBasis = ...

    # orthogonality test
    ...

## ROM L2 projection 

def project_L2(u_full,Phi,Nx,Ny):
    a = ...
    u_proj = ...
    
    return a, u_proj


mu = (0.99, 0.8, 0.2, 0.78)
...
xc, yc,M,b = VF.assemble_tpfa(Nx=50, Ny=50,mu=mu)
U = VF.solve_tpfa(M,b)
a,u_proj=project_L2(U,Phi,Nx,Ny)
u_proj=u_proj.reshape(50,50)

plt.figure()
plt.title("TPFA solution u_K (Dirichlet u=0 on ∂Ω)")
plt.imshow(u_proj.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
plt.colorbar()
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.show()


# POD-Galerkin
def solve_tpfa_rom(mu, Nx, Ny, Phi):
    _,_,A,l = assemble_tpfa(Nx=Ny, Ny=Ny, mu = mu)
    A_mu = Phi.transpose()@A@Phi 
    l_mu = Phi.transpose()@l
    u_rom = np.inv(A_mu)@l_mu
    U_rom = u_rom.reshape((Nx, Ny), order="F")
    return a, U_rom


mu = (0.6, 0.5, 0.2, 0.8)
...

# true sol

xc, yc,M,b = T.assemble_tpfa(Nx=50, Ny=50,mu=mu)
...


#### Convergence
""" 
#(u,v)_L2=sum_K|K| u_k v_k 

mu = (0.6, 0.5, 0.2, 0.8)


# construct uref:
_,_, M, b = assemble_tpfa(Nx=500, Ny=500,mu=mu)
Uref = solve_tpfa(M,b,500,500)
xc_ref = (np.arange(500) + 0.5) /500
yc_ref = (np.arange(500) + 0.5) /500
interp = RegularGridInterpolator((xc_ref,yc_ref), Uref)


err_true=[]
err_rom=[]

Ns = [...,...,...,...,...]  # choose grid sizes to test
for n in Ns:
    # true sol
    xc, yc, M, b = ...
    U = ...
    #POD_Galerkin 
    Phi=...
    _,Uproj=...

    #restrict uref on fine mesh
    points_new = np.column_stack((xc.flatten(), yc.flatten()))
    Uref_interp = interp(points_new).reshape(n, n)

    ## print l2 error
    abs_true_error = ...
    l2_true_error=...
    print(l2_true_error)
    abs_rom_error = ...
    l2_rom_error=...


    # L2 errors (list)
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
plt.loglog(hs, 1/(hs**2), "-", label=r"$h^2}$")
plt.loglog(hs, err_rom, "+-", label=r"$\|u_{ref}-u_N\|_{L^2}$")
plt.gca().invert_xaxis()  # optional: smaller h to the right
plt.xlabel(r"$h$")
plt.ylabel(r"$L^2$ error")
plt.grid(True, which="both")
plt.legend()
plt.show()

"""