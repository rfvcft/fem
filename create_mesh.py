import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.core as cfc
import numpy as np
from matplotlib import pyplot as plt

MARK_NYLON = 1
MARK_COPPER = 2
MARK_CONVECTION = 11
MARK_FLUX = 12
MARK_NOFLUX = 13

def create_mesh(draw=True):
    L = 5 * 10e-6
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
    for i in range(3, 12):
        g.spline([i, i+1], marker=MARK_CONVECTION)
    g.spline([12, 13], marker=MARK_NOFLUX)
    g.spline([13, 14], marker=MARK_NOFLUX)
    g.spline([14, 0], marker=MARK_NOFLUX)

    g.spline([1, COPPER_NBR_POINTS + 1])
    for i in range(1, NYLON_NBR_POINTS - 3):
        g.spline([COPPER_NBR_POINTS + i, COPPER_NBR_POINTS + i + 1])
    g.spline([19, 14])

    g.surface([0] + list(range(15, 21)) + [14], marker=MARK_NYLON) # Nylon
    g.surface(list(range(1, 14)) + list(reversed(range(15, 21))), marker=MARK_COPPER) # Copper

    if draw:
        cfv.draw_geometry(g)
    # cfv.showAndWait()

    mesh = cfm.GmshMesh(g)
    mesh.elType = 2
    mesh.dofsPerNode = 1
    mesh.elSizeFactor = 1

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    cfv.figure()

    # Draw the mesh.

    if draw:
        cfv.drawMesh(
            coords=coords,
            edof=edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True,
            title="Example 01"
                )
        cfv.showAndWait()
    return coords, edof, dofs, bdofs, elementmarkers

coords, edof, dofs, bdofs, elementmarkers = create_mesh(draw=True)

print(bdofs)
print(len(bdofs))

NELEM, NDOF = len(edof), len(dofs)
k_copper = 385
k_nylon = 0.26
alpha = 40

K = np.zeros((NDOF, NDOF))
fb = np.zeros((NDOF, 1))

ex, ey = cfc.coord_extract(edof, coords, dofs)

for i in np.arange(0, NELEM):
    if elementmarkers[i] == 1: # Nylon
        Ke = cfc.flw2te(ex[i], ey[i], [1], k_nylon*np.eye(2))
    else: # Copper
        Ke = cfc.flw2te(ex[i], ey[i], [1], k_copper*np.eye(2))
    cfc.assem(edof[i,:], K, Ke)
