import calfem.vis_mpl as cfv
import calfem.core as cfc
import calfem.utils as cfu
import numpy as np
from create_mesh import create_mesh, MARK_NYLON, MARK_FLUX, MARK_XCLAMP, MARK_YCLAMP, MARK_CLAMP
from stationary import stationary_solver

def vonMises_solver():
    _, K, f, ex, ey, coords, edof, dofs, _, elementmarkers = stationary_solver()
    NELEM, NDOF = len(edof), len(dofs)

    bcPresc = np.array([], 'i')
    a, r = cfc.solveq(K, f, bcPresc)

    ### von Mises stress
    T_inf = 18
    E_copper = 128
    E_nylon = 3
    v_copper = 0.36
    v_nylon = 0.39
    alpha_copper = 17.6 * 10**-6
    alpha_nylon = 80 * 10**-6
    NDOF_S = 2*NDOF
    edof_S = np.zeros((NELEM, 6), dtype=int)
    bdofs_S = dict()

    _, edof_S, dofs_S, bdofs_S, _ = create_mesh(draw=False, dofs_per_node=2, element_size_factor=0.007)

    def Dmatrix(E, v):
        return E/((1+v)*(1-2*v)) * np.array([[1-v, v, 0],
                                             [v, 1-v, 0],
                                             [0, 0, 0.5*(1-2*v)]])

    D_copper = Dmatrix(E_copper, v_copper)
    D_nylon = Dmatrix(E_nylon, v_nylon)

    Ks = np.zeros((NDOF_S, NDOF_S))
    fs0 = np.zeros((NDOF_S, 1))

    for i in np.arange(0, NELEM):
        deltaT = np.sum([a[edof[i][j] - 1] for j in range(3)])/3 - T_inf
        if elementmarkers[i] == MARK_NYLON: # Nylon
            Kse = cfc.plante(ex[i], ey[i], [2, 1], D_nylon)
            epsilon_dT = alpha_nylon * deltaT * np.array([[1], [1], [0]])
            fs0e = cfc.plantf(ex[i], ey[i], [2, 1], (D_nylon @ epsilon_dT).T)
        else: # Copper
            Kse = cfc.plante(ex[i], ey[i], [2, 1], D_copper)
            epsilon_dT = alpha_copper * deltaT * np.array([[1], [1], [0]])
            fs0e = cfc.plantf(ex[i], ey[i], [2, 1], (D_nylon @ epsilon_dT).T)
        for j in range(6):
            fs0[edof_S[i][j] - 1] += fs0e[j]
        cfc.assem(edof_S[i], Ks, Kse)

    bc_S = np.array([], 'i')
    bc_val_S = np.array([], 'f')
    bc_S, bc_val_S = cfu.apply_bc(bdofs_S, bc_S, bc_val_S, MARK_CLAMP)
    bc_S, bc_val_S = cfu.apply_bc(bdofs_S, bc_S, bc_val_S, MARK_FLUX)
    bc_S, bc_val_S = cfu.apply_bc(bdofs_S, bc_S, bc_val_S, MARK_XCLAMP, dimension=1)
    bc_S, bc_val_S = cfu.apply_bc(bdofs_S, bc_S, bc_val_S, MARK_YCLAMP, dimension=2)
    a, r = cfc.solveq(Ks, fs0, bc_S)

    ed = cfc.extract_eldisp(edof_S, a)
    vonMises = []

    for i in range(NELEM):
        if elementmarkers[i] == MARK_NYLON:
            es, et = cfc.plants(ex[i], ey[i], [2, 1], D_nylon, ed[i])
            E = E_nylon
            v = v_nylon
            alpha = alpha_nylon
        else:
            es, et = cfc.plants(ex[i], ey[i], [2, 1], D_copper, ed[i])
            E = E_copper
            v = v_copper
            alpha = alpha_copper
        sigx, sigy, tauxy = es[0]
        # epsx, epsy, gamxy = et[0]
        deltaT = np.mean([a[edof[i][j] - 1] for j in range(3)]) - T_inf
        sigx -= alpha*E*deltaT/(1-2*v)
        sigy -= alpha*E*deltaT/(1-2*v)
        # sigz = E*v/((1+v)*(1-2*v))*(epsx + epsy) - alpha*E/(1-2*v)*deltaT
        sigz = v*(sigx + sigy) - (alpha * E * deltaT)
        sigz -= alpha*E*deltaT/(1-2*v)
        vonMises.append(np.sqrt(sigx**2 + sigy**2 + sigz**2 - sigx*sigy - sigx*sigz - sigy*sigz + 3*tauxy**2))

    node_stresses = np.zeros((NDOF, 1))
    for i in range(NDOF):
        x = dofs_S[i][0]
        idxs = np.where(x == edof_S)[0]
        node_stresses[i] = np.mean([vonMises[idx] for idx in idxs])

    # cfv.figure((10,10))
    # cfv.draw_nodal_values_shaded(node_stresses, coords, edof)

    cfv.figure((10,10))
    magnification = 5
    L = 0.005
    flip_y = np.array([([1, -1]*int(a.size/2))]).T
    flip_x = np.array([([-1, 1]*int(a.size/2))]).T
    cfv.draw_element_values(vonMises, coords, edof_S, 2, 2, a,
                            draw_elements=False, draw_undisplaced_mesh=True,
                            title="Effective Stress", magnfac=magnification)
    cfv.draw_element_values(vonMises, [0, L]+[1, -1]*coords, edof_S, 2, 2, np.multiply(flip_y, a),
                            draw_elements=False, draw_undisplaced_mesh=True,
                            title="Effective Stress", magnfac=magnification)
    cfv.draw_element_values(vonMises, [2*L, L]+[-1, -1]*coords, edof_S, 2, 2, np.multiply(flip_y*flip_x, a),
                            draw_elements=False, draw_undisplaced_mesh=True,
                            title="Effective Stress", magnfac=magnification)
    cfv.draw_element_values(vonMises, [2*L, 0]+[-1, 1]*coords, edof_S, 2, 2, np.multiply(flip_x, a),
                            draw_elements=False, draw_undisplaced_mesh=True,
                            title="Effective Stress", magnfac=magnification)
    cfv.colorbar()
    # cfv.draw_displacements(a, coords, edof, 1, 2,
    #                        draw_undisplaced_mesh=True, title="Example 06 - Displacements",
    #                        magnfac=2)
    cfv.show_and_wait()

if __name__ == '__main__':
    vonMises_solver()
