# import packages
import numpy as np
import matplotlib.pyplot as plt

## Fonction de référence
delta = 0.001
def f(x,mu, delta) : 
    return np.tanh((x-mu)/delta)


def generate_translation_snapshots(Ne, n_mu,x0 = 0, x1 = 94, delta = delta):
    """
    Ne+1 mesh element95   mesh starting point x0
    mesh last point x1
    jump width: delta
    number of tested parameters n_mu<Ne
    """

    
    # fixed mesh
    x = np.linspace(x0, x1, Ne)

    # uniform mu
    # n_mu values between x0 and x1
    seed = np.random.seed(42)
    mu_values = np.linspace(x0, x1, n_mu)
    #mu_values =x0*np.eye(1,n_mu) + (x1-x0)*np.random.rand(1, n_mu)

    snapshots = np.zeros((Ne, n_mu))

    for i in range(mu_values.shape[0]):
        u = f(x, mu_values[i], delta)
        snapshots[:,i] = u
    
    return snapshots, x, mu_values # should return snapshots, mesh, mu_values


## Test

snapshots,x,_=generate_translation_snapshots(Ne=300, n_mu = 100)
for i in range(snapshots.shape[1]):
    plt.plot(x,snapshots[:,i])
plt.xlabel("x")
plt.ylabel("u(x)")
plt.tight_layout()
plt.show()


""" OFFLINE """
""" POD """

def Construct_RB(NumberOfSnapshots,Ne,NumberOfModes):
    
    """
    Number of training parameters: NumberOfSnapshots
    Mesh size : Ne=100
    Number of basis functions: NumberOfModes
    """
    
    Snapshots,x,MU = generate_translation_snapshots(Ne = Ne, n_mu = NumberOfSnapshots)
    print("Number of snapshots:", NumberOfSnapshots)
    
    volK = np.abs(x[0] - x[1]) 
    C = volK*Snapshots.T@Snapshots
    EigenValues, EigenVectors = np.linalg.eigh(C, UPLO="L") 
    ## Vecteur propre stocké en colonne

    idx = EigenValues.argsort()[::-1]
    TotEigenValues = EigenValues[idx] # Valeurs propres réordonnées
    TotEigenVectors = EigenVectors[:, idx] # Réordonnement selon les valeurs propres

    # retrieve N=NumberOfModes first eigenvalues
    EigenValues = np.array([TotEigenValues[i] for i in range(NumberOfModes)]) # On prend les NumberOfModes première valeurs
    EigenVectors = np.array([TotEigenVectors[:,i] for i in range(NumberOfModes)]).T # Les vecteurs propres associés

    print("eigenvalues: ",EigenValues)

    RIC = 1 - sum(EigenValues[i] for i in range(NumberOfModes))/sum(lbd for lbd in TotEigenValues) #must be close to 0
    print("Relativ Information Content (must be close to 0): ",RIC)

    ChangeOfBasisMatrix = np.zeros((NumberOfSnapshots, NumberOfModes))

    for j in range(NumberOfModes):
        ChangeOfBasisMatrix[:,j] = EigenVectors[:,j]/np.sqrt(EigenValues[j]) #/ normalization
    
    ReducedBasis = Snapshots@ChangeOfBasisMatrix 

    # orthogonality test
    #AL_Id = ReducedBasis.T@ReducedBasis
    #print(np.isclose((1/volK)*np.eye(AL_Id.shape[0]), AL_Id))

    return ReducedBasis,Snapshots,EigenValues, MU

## L2 projection 
NumberOfSnapshots = 300
Ne = 100
NumberOfModes = 80

ReducedBasis,Snapshots,eigvals,_=Construct_RB(NumberOfSnapshots = NumberOfSnapshots, Ne = Ne, NumberOfModes = NumberOfModes)

# POD eigenvalues
plt.figure(figsize=(7, 5))
plt.semilogy(np.arange(1, len(eigvals) + 1), eigvals, 'o-')
plt.xlabel("N")
plt.ylabel("POD eigenvalues")
plt.grid(True)
plt.show()


### test

ReducedBasis,Snapshots,eigvals,_=Construct_RB(NumberOfSnapshots = NumberOfSnapshots, Ne = Ne, NumberOfModes = 50)
x = np.linspace(-100, 100, Ne)
plt.plot(x,ReducedBasis[:,0])
plt.plot(x,ReducedBasis[:,1])
plt.xlabel("x")
plt.ylabel("Phi[x]")
plt.tight_layout()
plt.show()


def project_snapshot(snapshot, x, ReducedBasis):
    a = ReducedBasis.T @ snapshot
    u_proj = ReducedBasis @ a
    return np.abs(x[0] - x[1])*u_proj


def compute_kolmogorov_decay(Snapshots, x,ReducedBasis):
 
    """
    Snapshots : array of snapshots
    x: mesh
    ReducedBasis: basis functions
    """

    NumberOfSnapshots, Ndofs = Snapshots.shape
    NumberOfModes = ReducedBasis.shape[0]
    dx = np.abs(x[0] - x[1])
    #print(np.shape(ReducedBasis))
    
    dmax = np.zeros(NumberOfModes)

    for n in range(1, NumberOfModes + 1): #forloop over n= 1 ... N
        ReducedBasis_n = ReducedBasis[:,0:n] # shape == (DoFs, n)
        errs = []
        for i in range(NumberOfSnapshots): #for each snapshots
            u_proj = ReducedBasis_n@(ReducedBasis_n.T@Snapshots[:,i]) # Projection
            err = Snapshots[:,i] - u_proj
            err_l2 = np.sqrt(np.sum(err**2) )
            errs.append(err_l2)
        dmax[n - 1] = np.max(errs)

    return dmax


# -------------------------
# Test
# -------------------------
NumberOfSnapshots = 100
Ne = 100
NumberOfModes = 50

ReducedBasis,Snapshots,_,_= Construct_RB(NumberOfSnapshots=NumberOfSnapshots,Ne=Ne,NumberOfModes=NumberOfModes)

x0=-100
x1=100
x = np.linspace(x0, x1, Ne + 1)
dmax = compute_kolmogorov_decay(Snapshots,x, ReducedBasis)

# log-log estimation of the slope
nvals = np.arange(1, len(dmax) + 1)

#Use np.polyfit(x,y,deg) to fit a polynomial p[0] * x + p[1] to points (nvals, dmax ) in log scale
#
coef = np.polyfit(np.log(nvals), np.log(dmax), 1)
alpha = coef[0]
print(f"Slope ~ n^(alpha) : alpha = {alpha:.3f}")

# Plot
plt.figure(figsize=(7, 5))
plt.loglog(nvals, dmax, 'o-', label=r'$d_n^{\max}$')

plt.loglog(nvals,np.exp(coef[1]) * nvals**coef[0],  '--',label=fr'fit $\sim n^{{{alpha:.2f}}}$')
plt.xlabel("n")
plt.ylabel("Projection error")
plt.grid(True, which="both")
plt.legend()
plt.show()
