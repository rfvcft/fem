import calfem.vis_mpl as cfv
import calfem.core as cfc
import numpy as np
from matplotlib import pyplot as plt
from plantml import plantml
from create_mesh import MARK_NYLON
from stationary import stationary_solver

def transient_solver():
    a_stationary, K, f, ex, ey, coords, edof, dofs, bdofs, elementmarkers = stationary_solver()
    NELEM, NDOF = len(edof), len(dofs)

    a = 18 * np.ones((NDOF, 1))
    dt = 0.1
    tf = 2.4
    rho_nylon = 1100
    rho_copper = 8930
    c_nylon = 1500
    c_copper = 386
    stat_temp = np.max(a_stationary)
    C = np.zeros((NDOF, NDOF))
    for i in np.arange(0, NELEM):
        if elementmarkers[i] == MARK_NYLON: # Nylon
            Ce = plantml(ex[i], ey[i], rho_nylon * c_nylon)
        else: # Copper
            Ce = plantml(ex[i], ey[i], rho_copper * c_copper)
        cfc.assem(edof[i], C, Ce)

    j = 1
    for t in np.arange(0, tf, step=dt):
        a = np.linalg.solve(C + dt*K, C @ a + dt*f)
        if t > j * 0.44:
            plt.figure()
            cfv.draw_nodal_values_shaded(a, coords * 10**3, edof)
            cbar = cfv.colorbar()
            cbar.set_label('Temperature [°C]', rotation=90)
            plt.title(f"Time = {float(t):.1f} s")
            plt.xlabel('Length [mm]')
            plt.ylabel('Length [mm]')
            plt.set_cmap("inferno")
            j += 1
        if np.max(a) > 0.9 * stat_temp:
            print(t)
            break

    cfv.draw_nodal_values_shaded(a, coords, edof)
    cfv.colorbar()
    plt.set_cmap("inferno")

    cfv.show_and_wait()

if __name__ == '__main__':
    transient_solver()
