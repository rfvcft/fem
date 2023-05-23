import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

# Values for surface- and boundary-marks
MARK_NYLON = 11
MARK_COPPER = 12
MARK_CONVECTION = 1
MARK_FLUX = 2
MARK_CLAMP = 3
MARK_XCLAMP = 4
MARK_YCLAMP = 5

L = 5 * 10**-3

def define_geometry():
    a = 0.1*L
    b = 0.1*L
    c = 0.3*L
    d = 0.05*L
    h = 0.15*L
    t = 0.05*L

    g = cfg.Geometry()

    NYLON_NBR_POINTS = 8
    COPPER_NBR_POINTS = 14

    # Declare points
    g.point([0, 0])
    g.point([0, 0.5*L - a - b])
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

    # Declare edges (with marks where necessary)
    g.spline([0, 1], marker=MARK_CLAMP)
    g.spline([1, 2], marker=MARK_CLAMP)
    g.spline([2, 3], marker=MARK_FLUX)
    g.spline([3, 4], marker=MARK_YCLAMP)
    g.spline([4, 5], marker=MARK_CONVECTION)
    g.spline([5, 6], marker=MARK_CONVECTION)
    g.spline([6, 7], marker=MARK_CONVECTION)
    g.spline([7, 8], marker=MARK_CONVECTION)
    g.spline([8, 9], marker=MARK_CONVECTION)
    g.spline([9, 10], marker=MARK_CONVECTION)
    g.spline([10, 11], marker=MARK_XCLAMP)
    g.spline([11, 12], marker=MARK_CONVECTION)
    g.spline([12, 13], marker=MARK_CONVECTION)
    g.spline([13, 14])
    g.spline([14, 0])
    g.spline([1, COPPER_NBR_POINTS + 1])
    for i in range(1, NYLON_NBR_POINTS - 3):
        g.spline([COPPER_NBR_POINTS + i, COPPER_NBR_POINTS + i + 1])
    g.spline([19, 14])

    # Declare the two types of surfaces
    g.surface([0] + list(range(15, 21)) + [14], marker=MARK_NYLON)                      # Nylon
    g.surface(list(range(1, 14)) + list(reversed(range(15, 21))), marker=MARK_COPPER)   # Copper

    return g


def create_mesh(draw=True, dofs_per_node=1, element_size_factor = 0.03):
    g = define_geometry()
    mesh = cfm.GmshMesh(g)
    mesh.elType = 2
    mesh.dofsPerNode = dofs_per_node
    mesh.elSizeFactor = element_size_factor
    mesh.return_boundary_elements = True

    coords, edof, dofs, bdofs, elementmarkers, boundary_elements = mesh.create()

    # Draw the mesh.

    if draw:
        cfv.draw_geometry(g)
        cfv.drawMesh(
            coords=coords,
            edof=edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True,
            title="Mesh for gripper"
                )
        cfv.showAndWait()
    return coords, edof, dofs, bdofs, elementmarkers, boundary_elements
