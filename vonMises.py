import calfem.vis_mpl as cfv
import calfem.core as cfc
import calfem.utils as cfu
import numpy as np
from create_mesh import MARK_NYLON, MARK_FLUX, MARK_XCLAMP, MARK_YCLAMP, MARK_CLAMP, L
from stationary import stationary_solver

def vonMises_solver():
    a_stat, _, _, ex, ey, coords, edof, dofs, bdofs, elementmarkers = stationary_solver(element_size_factor=0.007)
    NELEM, NDOF = len(edof), len(dofs)

    T_inf = 18
    E_copper = 128 * 10**9
    E_nylon = 3 * 10**9
    v_copper = 0.36
    v_nylon = 0.39
    alpha_copper = 17.6 * 10**-6
    alpha_nylon = 80 * 10**-6

    # Create new dofs, edofs and bdofs as we now have two degrees of freedom for displacement
    NDOF_S = 2*NDOF

    # New dofs through the map n -> [2n-1, 2n]
    dofs_S = np.array([[2*n-1, 2*n] for n in range(1, NDOF+1)])

    # New edof
    edof_S = np.zeros((NELEM, 6), dtype=int)
    for i, row in enumerate(edof):
        edof_S[i] = [2*row[0]-1, 2*row[0], 2*row[1]-1, 2*row[1], 2*row[2]-1, 2*row[2]]

    # New bdofs
    bdofs_S = {}
    for mark, nodes in bdofs.items():
        bdofs_S[mark] = [dof for node in nodes for dof in [2*node-1, 2*node]]

    # Create our D-matrix as done in the "Termoelasticetet - plan töjning - uppdatering"-post on Canvas
    def Dmatrix(E, v):
        return E/((1+v)*(1-2*v)) * np.array([[1-v, v, 0],
                                             [v, 1-v, 0],
                                             [0, 0, 0.5*(1-2*v)]])

    D_copper = Dmatrix(E_copper, v_copper)
    D_nylon = Dmatrix(E_nylon, v_nylon)

    # Assemble the global stiffnes matrix for the stress problem
    Ks = np.zeros((NDOF_S, NDOF_S))
    fs0 = np.zeros((NDOF_S, 1))

    def create_element_values(ex, ey, alpha, D):
        Ke = cfc.plante(ex, ey, [2, 1], D)
        epsilon_dT = alpha * deltaT * np.array([[1], [1], [0]])
        fe = cfc.plantf(ex, ey, [2, 1], (D @ epsilon_dT).T)
        return Ke, fe

    for i in range(NELEM):
        deltaT = np.mean([a_stat[edof[i][j] - 1] for j in range(3)]) - T_inf # Average increase in temperature for an element
        if elementmarkers[i] == MARK_NYLON: # Nylon
            Kse, fs0e = create_element_values(ex[i], ey[i], alpha_nylon, D_nylon)
        else:                               # Copper
            Kse, fs0e = create_element_values(ex[i], ey[i], alpha_copper, D_copper)

        # Add the forces to the force vector f0
        for j in range(6):
            fs0[edof_S[i][j] - 1] += fs0e[j]

        cfc.assem(edof_S[i], Ks, Kse)

    # Apply boundary conditions
    bc_S = np.array([], 'i')
    bc_val_S = np.array([], 'f')
    bc_S, bc_val_S = cfu.apply_bc(bdofs_S, bc_S, bc_val_S, MARK_CLAMP)
    bc_S, bc_val_S = cfu.apply_bc(bdofs_S, bc_S, bc_val_S, MARK_FLUX)
    bc_S, bc_val_S = cfu.apply_bc(bdofs_S, bc_S, bc_val_S, MARK_XCLAMP, dimension=1) # Only allowed to move in y-direction
    bc_S, bc_val_S = cfu.apply_bc(bdofs_S, bc_S, bc_val_S, MARK_YCLAMP, dimension=2) # Only allowed to move in x-direction

    # Solve the FE-system
    a_S, _ = cfc.solveq(Ks, fs0, bc_S)

    # Create the von Mises stress
    ed = cfc.extract_eldisp(edof_S, a_S)
    vonMises = []
    for i in range(NELEM):
        if elementmarkers[i] == MARK_NYLON: # Nylon
            D_nylon = cfc.hooke(2, E_nylon, v_nylon)
            es, _ = cfc.plants(ex[i], ey[i], [2, 1], D_nylon, ed[i])
            E, v, alpha = E_nylon, v_nylon, alpha_nylon
        else:                               # Copper
            D_copper = cfc.hooke(2, E_copper, v_copper)
            es, _ = cfc.plants(ex[i], ey[i], [2, 1], D_copper, ed[i])
            E, v, alpha = E_copper, v_copper, alpha_copper
        sigx, sigy, sigz, tauxy = es[0]
        deltaT = np.mean([a_stat[edof[i][j] - 1] for j in range(3)]) - T_inf # Temperature increase

        # Factor in the stress from temperature increase
        k_T = alpha*E*deltaT/(1-2*v)
        sigx -= k_T
        sigy -= k_T
        sigz -= k_T
        stress = np.sqrt(sigx**2 + sigy**2 + sigz**2 - sigx*sigy - sigx*sigz - sigy*sigz + 3*tauxy**2)
        vonMises.append(stress)

    # Calculate the stress for each node by averaging the stress for all elements the node is included in
    node_stresses = np.zeros((NDOF, 1))
    for i in range(NDOF):
        x = dofs_S[i][0]
        idxs = np.where(x == edof_S)[0]
        node_stresses[i] = np.mean([vonMises[idx] for idx in idxs])

    cfv.figure((10,10))
    cfv.draw_nodal_values_shaded(node_stresses, coords, edof)

    # Draw the displacement of the whole gripper
    cfv.figure((10,10))
    magnification = 5
    flip_y = np.array([([1, -1]*int(a_S.size/2))]).T
    flip_x = np.array([([-1, 1]*int(a_S.size/2))]).T
    cfv.draw_element_values(vonMises, coords, edof_S, 2, 2, a_S,
                            draw_elements=False, draw_undisplaced_mesh=True,
                            title="Effective Stress, magnification=%ix" % magnification, magnfac=magnification)
    cfv.draw_element_values(vonMises, [0, L]+[1, -1]*coords, edof_S, 2, 2, np.multiply(flip_y, a_S),
                            draw_elements=False, draw_undisplaced_mesh=True, magnfac=magnification)
    cfv.draw_element_values(vonMises, [2*L, L]+[-1, -1]*coords, edof_S, 2, 2, np.multiply(flip_y*flip_x, a_S),
                            draw_elements=False, draw_undisplaced_mesh=True, magnfac=magnification)
    cfv.draw_element_values(vonMises, [2*L, 0]+[-1, 1]*coords, edof_S, 2, 2, np.multiply(flip_x, a_S),
                            draw_elements=False, draw_undisplaced_mesh=True, magnfac=magnification)
    cfv.colorbar()
    cfv.show_and_wait()

if __name__ == '__main__':
    vonMises_solver()
