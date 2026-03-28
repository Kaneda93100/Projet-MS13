import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tools as T
import time

Nx = 50; Ny = 20
mu = (0.99, 0.8, 0.2, 0.78)

xc, yc,M,b = T.assemble_tpfa(Nx=Nx, Ny=Ny,mu=mu)
U = T.solve_tpfa(M,b, Nx, Ny)

Phi = T.Construct_RB(NumberOfSnapshots= 5,Nx = Nx, Ny = Ny, NumberOfModes=3)
a,u_proj= T.project_L2(U,Phi,Nx,Ny)
u_proj=u_proj.reshape(Nx,Ny, order='F')

plt.figure()
plt.title("TPFA solution u_K (Dirichlet u=0 on ∂Ω)")
plt.imshow(u_proj.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
plt.colorbar()
plt.xlabel("x"); plt.ylabel("y") 
plt.tight_layout()
plt.show()


#### Performances

np.random.seed(42)
nbr_test = 50
mu_sample = np.random.rand(4, nbr_test)
mu = (0.99, 0.8, 0.2, 0.78)
Nx = 50; Ny = Nx

## Test TPFA
start_tpfa = time.perf_counter()

for i in range(50):
    _,_,M,b = T.assemble_tpfa(Nx = Nx, Ny = Ny, mu = mu_sample[:,i])
    U = T.solve_tpfa(M,b,Nx,Ny)

stop_tpfa = time.perf_counter()

print(f"\nTPFA : {stop_tpfa - start_tpfa}\n")

## Test ROM

start_rom = time.perf_counter()

Phi = T.Construct_RB(NumberOfSnapshots = 50, NumberOfModes = 3, Nx = Nx, Ny = Ny)


for i in range(50):
    _,_,M,b = T.assemble_tpfa(Nx = Nx, Ny = Ny, mu = mu)
    _,U_rom = T.solve_tpfa_rom(mu = mu_sample[:,i], Nx = Nx, Ny = Ny, Phi = Phi)

stop_rom = time.perf_counter()

print(f"\nROM : {stop_rom - start_rom}\n")

exit(-1)

#### Convergence

# Solution Volume finis

xc, yc,M,b = T.assemble_tpfa(Nx=Nx, Ny=Nx,mu=mu)
U_h = sp.sparse.linalg.spsolve(M,b)

#(u,v)_L2=sum_K|K| u_k v_k 

mu = (0.6, 0.5, 0.2, 0.8)


# construct uref on a fine grid so all test meshes stay within its domain
Nref = 60
_,_, M, b = T.assemble_tpfa(Nx=Nref, Ny=Nref, mu=mu)
Uref = T.solve_tpfa(M, b, Nref, Nref)
xc_ref = (np.arange(Nref) + 0.5) / Nref
yc_ref = (np.arange(Nref) + 0.5) / Nref
interp = sp.interpolate.RegularGridInterpolator((xc_ref, yc_ref), Uref.reshape(Nref, Nref, order='F'))


err_true=[]
err_rom=[]

Ns = [10, 20, 30, 40, 50]  # choose grid sizes to test
for n in Ns:
    # true sol
    xc, yc, M, b = T.assemble_tpfa(n, n, mu)
    U = T.solve_tpfa(M,b,n,n)
    #POD_Galerkin
    Phi= T.Construct_RB(50,n,n,10) # mettre 10 ici ?????????????????????????????????????
    _, U_rom= T.solve_tpfa_rom(mu, n, n, Phi)


    #restrict uref on fine mesh
    """Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    points_new = np.column_stack((Xc.flatten(order="F"), Yc.flatten(order="F")))"""
    points_new = np.array([[x,y] for x in xc for y in yc])


    Uref_interp = interp(points_new).reshape(n, n)

    ## print l2 error

    dx = 1./n
    dy = 1./n
    volK = dx*dy
    print("volK:",volK)

    abs_true_error = (U - Uref_interp)**2
    l2_true_error = np.sqrt(volK * np.sum(abs_true_error))
    print("l2 true error:", l2_true_error)
    #abs_rom_error = (U_rom - Uref_interp)**2
    abs_rom_error = (U_rom - Uref_interp)**2
    l2_rom_error = np.sqrt(volK * np.sum(abs_rom_error))
    print("l2_rom error:", l2_rom_error)

    # L2 errors (list)
    err_true.append(l2_true_error)
    err_rom.append(l2_rom_error)


# ---------------------------
# Plot log-log convergence
# ---------------------------
hs = np.array(Ns)
err_true = np.array(err_true)
err_rom = np.array(err_rom)

coefs = np.polyfit(np.log(hs), np.log(err_true), 1)
print(coefs[0])

plt.figure()
plt.loglog(hs, err_true, "o-", label=r"$\|u_{ref}-u\|_{L^2}$")
plt.loglog(hs, 1/(hs**2), "-", label=r"$h^2$")
plt.loglog(hs, err_rom, "+-", label=r"$\|u_{ref}-u_N\|_{L^2}$")
plt.gca().invert_xaxis()  # optional: smaller h to the right
plt.xlabel(r"$h$")
plt.ylabel(r"$L^2$ error")
plt.grid(True, which="both")
plt.legend()
plt.show()