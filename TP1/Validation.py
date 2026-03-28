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
Nx = 10; Ny = Nx

## Test TPFA

print("\n\nDébut du test TPFA\n\n")
start_tpfa = time.perf_counter()
for i in range(50):
    _,_,M,b = T.assemble_tpfa(Nx = Nx, Ny = Ny, mu = mu_sample[:,i])
    U = T.solve_tpfa(M,b,Nx,Ny)
stop_tpfa = time.perf_counter()

print(f"\nTPFA : {stop_tpfa - start_tpfa}\n")

## Test ROM

print("\n\nConstruction de la base réduite\n\n")
start_rb = time.perf_counter()
Phi = T.Construct_RB(NumberOfSnapshots = 50, NumberOfModes = 3, Nx = Nx, Ny = Ny)
stop_rb = time.perf_counter()
print(f"RB : {stop_rb - start_rb}\n")

print("\n\nDébut du test ROM\n\n")

start_rom = time.perf_counter()
for i in range(50):
    _,_,M,b = T.assemble_tpfa(Nx = Nx, Ny = Ny, mu = mu_sample[:,i])
    _,U_rom = T.solve_tpfa_rom(mu = mu_sample[:,i], Nx = Nx, Ny = Ny, Phi = Phi, A = M, l = b)
stop_rom = time.perf_counter()

print(f"\nROM : {stop_rom - start_rom}\n")


perf_hf = [0.2603350999997929, 0.6296214999965741, 1.123014199998579, 1.8335791999998037, 2.755822500002978, 6.196164299995871, 15.358156100002816,129.4399154999992]
perf_rom = [0.3095513000007486, 0.9868257999987691, 1.84800279999763, 2.930363399995258,  4.271860499997274, 4.3482680000015534, 9.526180599998042,65.14382860000478]
perf_rb = [0.3891774000003352, 0.9403516999955173, 1.7942659000036656, 2.8046219000025303, 4.7663010000032955, 6.613594000002195, 17.091368100002, 130.68198700000357]
x = [30, 50, 70, 90, 110, 130, 200, 500]


plt.figure()
plt.plot(x,perf_rom, "o-", label="ROM")
plt.plot(x,perf_hf, "s-", label="Haute fidélité")
#plt.plot(x,perf_rb, "^-", label="Base réduite")
plt.xlabel("N")
plt.ylabel("Temps (s)")
plt.title("Comparaison des performances")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x, perf_rb, "o-", label = "Base réduite")
plt.plot(x, perf_hf, "s-", label = "HF")
plt.xlabel("N")
plt.ylabel("Temps (s)")
plt.title("Temps de construction de la base réduite contre résolution HF")
plt.legend()
plt.grid(True)
plt.show()

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