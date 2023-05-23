import calfem.vis_mpl as cfv
import calfem.core as cfc
import numpy as np
from matplotlib import pyplot as plt
from plantml import plantml
from create_mesh import MARK_NYLON
from stationary import stationary_solver

def transient_solver():
    # First solve the stationary problem, and access the topology variables as well as K and f
    a_stationary, K, f, ex, ey, coords, edof, dofs, _, elementmarkers = stationary_solver(element_size_factor=0.03)
    NELEM, NDOF = len(edof), len(dofs)

    # Initial temperature
    a = 18 * np.ones((NDOF, 1))
    # Time step
    dt = 0.1
    tf = 100
    # Material constants
    rho_nylon = 1100
    rho_copper = 8930
    c_nylon = 1500
    c_copper = 386
    stat_temp = np.max(a_stationary)

    # Assemble the C-matrix
    C = np.zeros((NDOF, NDOF))
    for i in np.arange(0, NELEM):
        if elementmarkers[i] == MARK_NYLON: # Nylon
            Ce = plantml(ex[i], ey[i], rho_nylon * c_nylon)
        else:                               # Copper
            Ce = plantml(ex[i], ey[i], rho_copper * c_copper)
        cfc.assem(edof[i], C, Ce)

    j = 1
    for t in np.arange(0, tf, step=dt):
        # Implicit Euler time step
        a = np.linalg.solve(C + dt*K, C @ a + dt*f)
        if t > j * 0.44 and j <= 5: # Draw 5 pictures under the first 3% of the time under 90%
                                    # 0.44 was calculated to satisfy this
            plt.figure()
            cfv.draw_nodal_values_shaded(a, coords * 10**3, edof)
            cbar = cfv.colorbar()
            cbar.set_label('Temperature [°C]', rotation=90)
            plt.clim(18, 28) # Force colorbar limits to be the same
            plt.title(f"Time = {float(t):.1f} s")
            plt.xlabel('Length [mm]')
            plt.ylabel('Length [mm]')
            plt.set_cmap("inferno")
            j += 1
        if np.max(a) > 0.9 * stat_temp:
            break

    plt.figure()
    cfv.draw_nodal_values_shaded(a, coords * 10**3, edof)
    cbar = cfv.colorbar()
    cbar.set_label('Temperature [°C]', rotation=90)
    plt.title(f"Time = {float(t):.1f} s, 90% of stationary max temperature")
    plt.xlabel('Length [mm]')
    plt.ylabel('Length [mm]')
    plt.set_cmap("inferno")
    cfv.show_and_wait()

if __name__ == '__main__':
    transient_solver()
