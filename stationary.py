import calfem.vis_mpl as cfv
import calfem.core as cfc
import numpy as np
from matplotlib import pyplot as plt
from math import dist
from create_mesh import create_mesh, MARK_NYLON, MARK_COPPER, MARK_CONVECTION, MARK_FLUX

def stationary_solver(draw_mesh=False, draw=False):
    coords, edof, dofs, bdofs, elementmarkers = create_mesh(draw=draw_mesh, element_size_factor=0.007)

    NELEM, NDOF = len(edof), len(dofs)
    k_copper = 385
    k_nylon = 0.26
    alpha_c = 40
    h = 10**5
    T_inf = 18


    K = np.zeros((NDOF, NDOF))
    fb = np.zeros((NDOF, 1))

    ex, ey = cfc.coord_extract(edof, coords, dofs)
    for i in np.arange(0, NELEM):
        if elementmarkers[i] == MARK_NYLON: # Nylon
            Ke = cfc.flw2te(ex[i], ey[i], [1], k_nylon * np.eye(2))
        if elementmarkers[i] == MARK_COPPER: # Copper
            Ke = cfc.flw2te(ex[i], ey[i], [1], k_copper*np.eye(2))
        cfc.assem(edof[i,:], K, Ke)


    for element in edof:
            in_boundary_qn = [False, False, False]
            in_boundary_qh = [False, False, False]
            for i in range(3):
                if element[i] in bdofs[MARK_CONVECTION]:
                    in_boundary_qn[i] = True
                if element[i] in bdofs[MARK_FLUX]:
                    in_boundary_qh[i] = True
            for i in range(2):
                for j in range(i + 1, 3):
                    if in_boundary_qn[i] and in_boundary_qn[j]:
                        Le = dist(coords[element[i] - 1], coords[element[j] - 1])
                        Kce = alpha_c*Le/6 * np.array([[2, 1], [1, 2]])
                        fb[element[i]-1] += alpha_c*Le*T_inf/2
                        fb[element[j]-1] += alpha_c*Le*T_inf/2
                        cfc.assem(np.array([element[i], element[j]]), K, Kce)
                    if in_boundary_qh[i] and in_boundary_qh[j]:
                        Le = dist(coords[element[i] - 1], coords[element[j] - 1])
                        fb[element[i]-1] += h*Le/2
                        fb[element[j]-1] += h*Le/2



    bcPresc = np.array([], 'i')
    a, _ = cfc.solveq(K, fb, bcPresc)

    if draw:
        cfv.draw_nodal_values_shaded(a, coords*10**3, edof)
        cbar = cfv.colorbar()
        cbar.set_label('Temperature [°C]', rotation=90)
        plt.xlabel('Length [mm]')
        plt.ylabel('Length [mm]')
        plt.set_cmap("inferno")
        cfv.show_and_wait()

    return a, K, fb, ex, ey, coords, edof, dofs, bdofs, elementmarkers

if __name__ == '__main__':
    stationary_solver(draw=True)
