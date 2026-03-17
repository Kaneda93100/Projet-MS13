import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tools as T


Nx = 50; Ny = 50
mu = (0.99, 0.8, 0.2, 0.78)
...
xc, yc,M,b = T.assemble_tpfa(Nx=Nx, Ny=Ny,mu=mu)
U = T.solve_tpfa(M,b, Nx, Ny)

Phi = T.Construct_RB(NumberOfSnapshots= 5,Nx = Nx, Ny = Ny, NumberOfModes=3)
a,u_proj= T.project_L2(U,Phi,Nx,Ny)
u_proj=u_proj.reshape(Nx,Ny)

plt.figure()
plt.title("TPFA solution u_K (Dirichlet u=0 on ∂Ω)")
plt.imshow(u_proj.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
plt.colorbar()
plt.xlabel("x"); plt.ylabel("y") 
plt.tight_layout()
plt.show()

#### Convergence

# Solution Volume finis

xc, yc,M,b = T.assemble_tpfa(Nx=Nx, Ny=Nx,mu=mu)
U_h = sp.sparse.linalg.spsolve(M,b)

#(u,v)_L2=sum_K|K| u_k v_k 

mu = (0.6, 0.5, 0.2, 0.8)


# construct uref:
_,_, M, b = T.assemble_tpfa(Nx=Nx, Ny=Ny,mu=mu)
Uref = T.solve_tpfa(M,b,Nx,Ny)
xc_ref = (np.arange(Nx) + 0.5) /Nx
yc_ref = (np.arange(Ny) + 0.5) /Ny
interp = sp.interpolate.RegularGridInterpolator((xc_ref,yc_ref), Uref)


err_true=[]
err_rom=[]

Ns = [10, 20, 30, 40, 50]  # choose grid sizes to test
for n in Ns:
    # true sol
    xc, yc, M, b = T.assemble_tpfa(n,n,mu)
    U = T.solve_tpfa(M,b,n,n)
    #POD_Galerkin 
    Phi= T.Construct_RB(Nx = n, Ny = n)
    _,Uproj= T.project_L2(U, Phi, n, n)

    #restrict uref on fine mesh
    points_new = np.column_stack((xc.flatten(), yc.flatten()))
    Uref_interp = interp(points_new)#.reshape(n, n)

    ## print l2 error
    abs_true_error = sum(np.abs(Uref_interp[i]-U_h[i]) for i in range(Uref_interp.shape[0]))
    l2_true_error = (1/n**2)*np.sqrt(sum((Uref_interp[i]-U_h[i])**2 for i in range(Uref_interp.shape[0])))
    print(l2_true_error)
    abs_rom_error = sum(np.abs(Uref_interp[i] - Uproj[i]) for i in range(Uref_interp.shape[0]))
    l2_rom_error = (1/n**2) * np.sqrt(sum((Uref_interp[i] - Uproj[i])**2 for i in range(Uref_interp.shape[0])))


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
plt.loglog(hs, 1/(hs**2), "-", label=r"$h^2$")
plt.loglog(hs, err_rom, "+-", label=r"$\|u_{ref}-u_N\|_{L^2}$")
plt.gca().invert_xaxis()  # optional: smaller h to the right
plt.xlabel(r"$h$")
plt.ylabel(r"$L^2$ error")
plt.grid(True, which="both")
plt.legend()
plt.show()