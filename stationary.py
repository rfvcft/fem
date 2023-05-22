import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.core as cfc
import numpy as np
from matplotlib import pyplot as plt
from math import dist

MARK_NYLON = 11
MARK_COPPER = 12
MARK_CONVECTION = 1
MARK_FLUX = 2
MARK_NOFLUX = 3

def create_mesh(draw=True):
    L = 5 * 0.001
    a = 0.1*L
    b = 0.1*L
    c = 0.3*L
    d = 0.05*L
    h = 0.15*L
    t = 0.05*L

    g = cfg.Geometry()

    # Nylon element
    NYLON_NBR_POINTS = 8
    g.point([0, 0])
    g.point([0, 0.5*L - a - b])

    # Copper element
    COPPER_NBR_POINTS = 14
    g.point([0, 0.5*L - b])
    g.point([0, 0.5*L])
    g.point([a, 0.5*L])
    g.point([a, 0.5*L - b])
    g.point([a + c, 0.5*L - b])
    g.point([a + c + d, 0.5*L - b - d])
    g.point([a + c + d, d])
    g.point([L - 2*d, c])
    g.point([L, c])
    g.point([L, c - d])
    g.point([L - 2*d, c - d])
    g.point([a + c + d, 0])
    g.point([c + d, 0])

    g.point([a, 0.5*L - a - b])
    g.point([a, 0.5*L - a - b - h])
    g.point([a + t, 0.5*L - a - b - h])
    g.point([a + t, 0.5*L - a - b])
    g.point([c + d, 0.5*L - a - b])

    # for i in range(COPPER_NBR_POINTS):
    #     g.spline([i, i+1])
    # g.spline([COPPER_NBR_POINTS, 0])
    g.spline([0, 1], marker=MARK_NOFLUX)
    g.spline([1, 2], marker=MARK_NOFLUX)
    g.spline([2, 3], marker=MARK_FLUX)
    g.spline([3, 4], marker=MARK_NOFLUX)
    # for i in range(4,9):
    #     g.spline([i, i+1], marker=MARK_CONVECTION)
    g.spline([4, 5], marker=MARK_CONVECTION)
    g.spline([5, 6], marker=MARK_CONVECTION)
    g.spline([6, 7], marker=MARK_CONVECTION)
    g.spline([7, 8], marker=MARK_CONVECTION)
    g.spline([8, 9], marker=MARK_CONVECTION)
    g.spline([9, 10], marker=MARK_CONVECTION)
    g.spline([10, 11], marker=MARK_NOFLUX)
    g.spline([11, 12], marker=MARK_CONVECTION)
    g.spline([12, 13], marker=MARK_CONVECTION)
    g.spline([13, 14], marker=MARK_NOFLUX)
    g.spline([14, 0], marker=MARK_NOFLUX)

    g.spline([1, COPPER_NBR_POINTS + 1])
    for i in range(1, NYLON_NBR_POINTS - 3):
        g.spline([COPPER_NBR_POINTS + i, COPPER_NBR_POINTS + i + 1])
    g.spline([19, 14])

    g.surface([0] + list(range(15, 21)) + [14], marker=MARK_NYLON) # Nylon
    g.surface(list(range(1, 14)) + list(reversed(range(15, 21))), marker=MARK_COPPER) # Copper

    # cfv.showAndWait()

    mesh = cfm.GmshMesh(g)
    mesh.elType = 2
    mesh.dofsPerNode = 1
    mesh.elSizeFactor = 0.03

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()


    # Draw the mesh.

    if draw:
        # cfv.figure()
        cfv.draw_geometry(g)
        # cfv.figure()
        cfv.drawMesh(
            coords=coords,
            edof=edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True,
            title="Mesh"
                )
        cfv.showAndWait()
    return coords, edof, dofs, bdofs, elementmarkers

coords, edof, dofs, bdofs, elementmarkers = create_mesh(draw=True)

NELEM, NDOF = len(edof), len(dofs)
k_copper = 385
k_nylon = 0.26
c_nylon = 1500
c_copper = 386
alpha_c = 40
h = 100000
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
                    Kce = alpha_c*Le/6*np.array([[2, 1], [1, 2]])
                    fb[element[i]-1] += alpha_c*Le*T_inf/2
                    fb[element[j]-1] += alpha_c*Le*T_inf/2
                    cfc.assem(np.array([element[i], element[j]]), K, Kce)
                if in_boundary_qh[i] and in_boundary_qh[j]:
                    Le = dist(coords[element[i] - 1], coords[element[j] - 1])
                    fb[element[i]-1] += h*Le/2
                    fb[element[j]-1] += h*Le/2



bcPresc = np.array([], 'i')
a, r = cfc.solveq(K, fb, bcPresc)
cfv.draw_nodal_values_shaded(a, coords*10**3, edof)
cbar = cfv.colorbar()
cbar.set_label('Temperature [°C]', rotation=90)
plt.xlabel('Length [mm]')
plt.ylabel('Length [mm]')
plt.set_cmap("inferno")
cfv.show_and_wait()
