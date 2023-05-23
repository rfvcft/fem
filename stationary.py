import calfem.vis_mpl as cfv
import calfem.core as cfc
import numpy as np
from matplotlib import pyplot as plt
from math import dist
from create_mesh import create_mesh, MARK_NYLON, MARK_COPPER, MARK_CONVECTION, MARK_FLUX

def stationary_solver(draw_mesh=False, draw=False):
    coords, edof, dofs, bdofs, elementmarkers, boundary_elements = create_mesh(draw=draw_mesh, element_size_factor=0.007)
    # print(boundary_elements)
    # print(boundary_elements[3])
    # print(boundary_elements[3][0]['node-number-list'])

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

    nodes_conv = [node['node-number-list'] for node in boundary_elements[MARK_CONVECTION]]
    nodes_flux = [node['node-number-list'] for node in boundary_elements[MARK_FLUX]]

    for node in nodes_conv:
        Le = dist(coords[node[0] - 1], coords[node[1] - 1])
        Kce = alpha_c*Le/6 * np.array([[2, 1], [1, 2]])
        fb[node[0]-1] += alpha_c*Le*T_inf/2
        fb[node[1]-1] += alpha_c*Le*T_inf/2
        cfc.assem(np.array([node[0], node[1]]), K, Kce)

    for node in nodes_flux:
        Le = dist(coords[node[0] - 1], coords[node[1] - 1])
        fb[node[0]-1] += h*Le/2
        fb[node[1]-1] += h*Le/2

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
