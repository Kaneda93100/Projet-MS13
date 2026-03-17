import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import random

"""
Implémentation de la méthode des volumes finis.
"""

def A_fct(x, y, mu1, mu2):
    # Diffusion parameter
    return 2*mu1 + mu2*np.sin(x+y)*np.cos(x*y)

def f_fct(x, y, mu3, mu4):
    # Right-hand side term
    return mu3*(1-y) + mu4*x*(1-x)

def tau_sigma_int(AK, AL, sigma_len, dK, dL): 
    # AK -> diffusion moyennée sur la cellule K
    # AL -> diffusion moyennée sur la cellule L (voisine de K)
    # sigma_len -> longueur de l'arête partagée par K et L dénommée "sigma"
    # dK -> distance de x_K (centre de K) à l'arête sigma
    # dL -> distance de x_L à l'arête sigma

    """
    Calculer le coeffcient pour le flux lorsque l'arête n'est pas sur le bord du domaine.
    """

    return sigma_len* AK * AL / (AL*dK + AK*dL)

def tau_sigma_ext(AK, sigma_len, dK):
    # tau_sigma = |sigma| * AK / dK
    return sigma_len * AK / dK

##### TPFA assembling #########################################
def assemble_tpfa(Nx=5, Ny=2, mu=(0.99, 0.8, 0.2, 0.78)):
    """
        Fonction de calcul des volumes finis pour un jeu de paramètre avec les fonctions définis en amont du code. 

        input : 
            - Nx --> Nombre de noeuds sur l'axe des abscisses
            - Ny --> Nombre de noeuds sur l'axe des ordonnées
            - mi --> Jeu de paramètres pour lequel on résout le problème

        output : 
            - Xc --> Coordonnée en abscisse des centres de chaque cellule
            - Yc --> Coordonnée en ordonnée des centres de chaque cellule
            - M  --> Matrice du système VF à résoudre pour obtenir la solution
            - b  --> Terme source discrétisé 
    """
    mu1, mu2, mu3, mu4 = mu
    dx, dy = 1/Nx , 1/Ny
    volK = dx*dy # Aire de la cellule

    # Centre des cellules
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy

    Xc, Yc = np.meshgrid(xc, yc, indexing="ij") # like matrix indexing
    
    # Fields sampled at x_K
    A = np.array([[A_fct(xc[i], yc[j], mu1, mu2) for j in range(Ny)] for i in range(Nx)])
    f = np.array([[f_fct(xc[i], yc[j], mu3, mu4) for j in range(Ny)] for i in range(Nx)]) 

    
    # Indexing K in T_h=(i,j) -> k (Indexation des cellules)
    def idK(i, j):  
        return i + Nx * j

    b = (f * volK).reshape(-1, order="F")  # ∫_K f ≈ f(x_K)|K| with same indexing (i + Nx*j)

    # Face geometry on Cartesian grid
    
    # vertical faces (E/W), d_{K,sigma} 
    # horizontal faces (N/S), d_{K,sigma} 
    sigma_len_EW, d_EW = dy , 0.5*dx  # longueur de la face verticale du rectangle, coordonnée y du milieu
    sigma_len_NS, d_NS = dx, 0.5*dy # longueur de la face horizontale du rectable, coordonnée x du milieu

    # ---------------------------------------
    # Assemble by iterating over faces sigma 
    # ---------------------------------------
    rows, cols, data = [], [], []
    
    # 1) Interior faces F_int: add tau_sigma(u_K - u_L) to eq(K), and symmetric to eq(L)
    # 1a) Vertical interior faces between K=(i,j) and L=(i+1,j)
    for j in range(Ny): #Ny vertical edges
        for i in range(Nx - 1): #Nx-2 vertical edges without external boundaries
            K = (i, j)
            L = (i+1, j) # right neighbor  !! Attention à la segfault !!
            AK = A[K]
            AL = A[L]

            kK = idK(i, j)
            kL = idK(i+1, j)
            
            tau = tau_sigma_int(AK, AL, sigma_len_EW, d_EW, d_EW)

            # Equation for K: tau*(u_K - u_L)
            rows += [kK, kK] #line K,K 
            cols += [kK, kL] # col K,L (right neighbor)
            data += [tau, -tau] #[K,L]

            # Equation for L: +tau*(u_L - u_K)
            rows += [kL, kL] # symetric
            cols += [kL, kK] # left neighbor
            data += [tau, -tau]

    # 1b) Horizontal interior faces between K=(i,j) and L=(i,j+1)
    for i in range(Nx): # For Ny=2, j=0 one edge S/N 
        for j in range(Ny-1): # For Nx=5, 0....Nx-1 horizontal edge
            K = (i,j)
            L = (i , j + 1)

            kK = idK(i, j)
            kL = idK(i,j+1)

            AK = A[K]
            AL = A[L]
            tau = tau_sigma_int(AK,AL,sigma_len_NS,d_NS,d_NS)

            rows += [kK , kL]
            cols += [kK , kK]
            data += [tau , -tau]

            rows += [kL , kK]
            cols += [kL , kL]
            data += [tau , -tau]

    # 2) Boundary faces F_ext (Dirichlet u=0): add tau_sigma * u_K to eq(K)
    # Left boundary (west faces) : i=0
    for j in range(Ny): # j=0,1
        K = (0,j)
        kK = idK(0,j)
        tau = tau_sigma_ext(A[K],sigma_len_EW,d_EW)
        rows.append(kK); cols.append(kK); data.append(tau)

    # Right boundary (east faces) : i=Nx-1 =4
    for j in range(Ny): #0,1
        K = (Nx-1, j) 
        kK = idK(Nx-1, j)
        tau = tau_sigma_ext(A[K], sigma_len_EW, d_EW)
        rows.append(kK); cols.append(kK); data.append(tau)

    # Bottom boundary (south faces) : j=0 
    for i in range(Nx):#0,1,2,3,4
        K = (i,0)
        kK = idK(i,0)
        tau = tau_sigma_ext(A[K], sigma_len_NS, d_NS)
        rows.append(kK); cols.append(kK); data.append(tau)

    # Top boundary (north faces) : j=Ny-1
    for i in range(Nx):
        K = (i,Ny-1)
        kK = idK(i,Ny-1)
        tau = tau_sigma_ext(A[K], sigma_len_NS, d_NS)
        rows.append(kK); cols.append(kK); data.append(tau)

    # Build sparse system and solve
    N=Nx*Ny
    M = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    
    return xc,yc,M,b

def solve_tpfa(M,b,Nx,Ny):
    # Solve Mu=b
    u = spla.spsolve(M, b)
    # Back to (Nx,Ny) array with u_K at cell centers
    U = u.reshape((Nx, Ny), order="F")
    return U 


def Construct_RB(NumberOfSnapshots=50,Nx=50,Ny=50,NumberOfModes=10):
    """
        Fonction qui calcule la POD d'un ensemble de solution à une EDP paramétrée. 
        On effectue une réduction sur le nombre de paramètre en s'appuyant sur la décomposition en valeur
        singulière de la matrice de corrélation de la SVD. 

        input : 
            - NumberOfSnapshots --> Nombre de solution calculée par VF pour un jeu de paramètre
            - Nx, Ny            --> Maillage du carré (on met Nx noeud sur l'abscisse et Ny sur l'ordonnée)
            - NumberOfModes     --> Nombre de dimension que l'on souhaite garder
        
        output : 
            - ReducedBasis      --> Base réduite à NumberOfModes colonnes de la matrice des snapshots
    """

    print("number of modes: ",NumberOfModes)
    mu = (0.99, 0.8, 0.2, 0.78) 

    Snapshots=np.zeros((Nx*Ny,NumberOfSnapshots))
    #---------------------------------#
    #      Generate the snapshots     #
    #---------------------------------#
    for i in range(NumberOfSnapshots):
        xc, yc,M,b = assemble_tpfa(Nx,Ny, mu= mu)
        Snapshots[:,i] = solve_tpfa(M,b,Nx,Ny).flatten(order='F')
        mu = [(random.uniform(0.0,1.0)) for i in range(4)] #random coefficients in [0, 1] 
    
    #---------------------------------#
    #      POD                        #
    #---------------------------------#

    #(u,v)_L2=sum_K|K| u_k v_k 
    volK = 1/Nx * 1/Ny #|K| --> Aire d'une cellule rectangulaire de longueur 1/Nx et de largeur 1/Ny

    #  snapshot correlation matrix C_ij = (u_i,u_j)
    CorrelationMatrix = volK*Snapshots.transpose()@Snapshots    

    # Then, we compute the eigenvalues/eigenvectors of C (EigenVectors=alpha)
    EigenValues, EigenVectors = np.linalg.eigh(CorrelationMatrix, UPLO="L") #SVD: C eigenVectors=eigenValues eigenVectors
    ## Vecteur propre stocké en colonne

    idx = EigenValues.argsort()[::-1] # sort the eigenvalues
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



    print("La base retournée est bien orthonormale !\n")
    return ReducedBasis


## ROM L2 projection 
def project_L2(u_full,Phi,Nx,Ny):
    u_full = u_full.flatten(order = 'F')
    a = 1/Nx * 1/Ny * Phi.T@u_full
    u_proj = Phi@a
    
    return a, u_proj


def solve_tpfa_rom(mu, Nx, Ny, Phi): ## Rom --> Reduced Order Model
    _,_,A,l = assemble_tpfa(Nx=Ny, Ny=Ny, mu = mu)
    
    A_mu = Phi.transpose()@A@Phi 
    l_mu = Phi.transpose()@l

    u_rom = np.linalg.inv(A_mu)@l_mu
    U_rom = (Phi@u_rom).reshape((Nx, Ny), order="F")

    return u_rom, U_rom