#!/usr/bin/env python
# coding: utf-8

# # Solving the Poisson equation for Newtonian Potential in 2D
# 
# We begin with Poisson's equation for the Newtonian gravitational potential $\Phi(x)$ of a uniform density disc in 2 dimensions, which reads 
# $$-\Delta\Phi(x) = \rho(x)\quad x\in S^d_R,\quad d=2.$$
# 
# Here, $S^d_R$ is the sphere of radius $R$ in dimension $d$, and $\rho(x)$ is the mass density, taken to be constant, $\rho(x) = \rho_0$ in what follows. 
# 
# The problem is particularly useful for testing numerical methods since in 2 and 3 dimensions it has an analytic solution [See Cohl and Palmer, *Fourier and Gegenbauer Expansions for a Fundamental Solution of Laplace's Equation in Hyperspherical Geometry*] 
# 
# In particular, for uniform density $\rho = \rho_0>0$, and radius $R = r_0>0$, the exact solution is
# 
# $$ \Phi(x) :=\cases{-\frac{\rho_0}{4}\big(r^2 - r_0^2 +2r_0^2 \log r_0\big),\quad r\in[0,r_0], \\
# -\frac{1}{2}\rho_0 r_0^2 \log r,\quad r\in(r_0,\infty).}$$
# 
# This formula says that $\Phi(r_0)=-\frac{1}{2}\rho_0 r_0^2 \log(r_0)$ at-and-beyond the boundary. We use this as the boundary data for our numerical solver. 

# In[1]:


########################################################################################################
################################## MESHING #############################################################
########################################################################################################
import gmsh
gmsh.initialize()

#we make a circular disc mesh by calling the addEllipse function with major and minor axes = 1
ellipse = gmsh.model.occ.addEllipse(0, 0, 0, 1, 1)
gmsh.model.occ.addCurveLoop([ellipse], 5)
membrane = gmsh.model.occ.addPlaneSurface([5])
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin",0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",0.05)
gmsh.model.mesh.generate(gdim)

from dolfinx.io import gmshio
from mpi4py import MPI

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

########################################################################################################
################################## DEFINE EXACT SOLUTION ###############################################
########################################################################################################
from dolfinx import fem, mesh, fem, io, nls, log
import ufl
import numpy
from petsc4py.PETSc import ScalarType

#define the values of mass density, radius, and log of radius, and create constant functions on 
#the mesh for these values
rho0_0 = 12 #Mass density
R0_0 = 1.0 #Radius of disc
lnR0_0 = numpy.log(R0_0) #log of disc radius

x = ufl.SpatialCoordinate(domain)
rho0 = fem.Constant(domain, ScalarType(rho0_0))
R0 = fem.Constant(domain, ScalarType(R0_0)) 
lnR0 = fem.Constant(domain, ScalarType(lnR0_0))

#p is the exact analytic solution
p = -.25 * rho0 * ((x[0]**2 + x[1]**2) - R0**2 + 2 * R0**2 * lnR0)

########################################################################################################
################################## DEFINE BOUNDARY CONDITION ###########################################
########################################################################################################
#note that u_ufl is the same function as p, but is specified using, e.g., R0_0 instead of R0
u_ufl = -.25 * rho0_0 * ((x[0]**2 + x[1]**2) - R0_0**2 + 2 * R0_0**2 * lnR0_0)
V = fem.FunctionSpace(domain, ("CG", 1))
u_exact = lambda x: eval(str(u_ufl))
u_D = fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool))

bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

########################################################################################################
################################## ALTERNATIVE SPECIFICATION OF BOUNDARY DATA ##########################
########################################################################################################
#import numpy as np
#def on_boundary(x):
#    return np.isclose(np.sqrt((x[0])**2 + x[1]**2), 1) #must be changed accordingly for elliptic boundary
#boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)

#bc = fem.dirichletbc(p, boundary_dofs, V)

########################################################################################################
################################## SET UP AND SOLVE VARIATIONAL PROBLEM ################################
########################################################################################################
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = rho0 * v * ufl.dx
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

########################################################################################################
################################## PLOT SOLUTION USING PYVISTA #########################################
########################################################################################################
from dolfinx.plot import create_vtk_mesh
import pyvista
pyvista.set_jupyter_backend("pythreejs")

# Extract topology from mesh and create pyvista mesh
topology, cell_types, x = create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)

# Set deflection values and add it to plotter
grid.point_data["u"] = uh.x.array
warped = grid.warp_by_scalar("u", factor=.5)

plotter = pyvista.Plotter()
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    pyvista.start_xvfb()
    plotter.screenshot("deflection.png")


# **Plot of analytical solution** 

# In[2]:


#pyvista.set_jupyter_backend("ipygany")
Q = fem.FunctionSpace(domain, ("CG", 5))
expr = fem.Expression(p, Q.element.interpolation_points())
pressure = fem.Function(Q)
pressure.interpolate(expr)

load_plotter = pyvista.Plotter()
p_grid = pyvista.UnstructuredGrid(*create_vtk_mesh(Q))
p_grid.point_data["p"] = pressure.x.array.real
warped_p = p_grid.warp_by_scalar("p", factor=0.5)
warped_p.set_active_scalars("p")
load_plotter.add_mesh(warped_p, show_scalar_bar=True)
load_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    load_plotter.show()
else:
    pyvista.start_xvfb()
    load_plotter.screenshot("load.png")


# Finally, compute the error between the numerical and analytic solution in a few norms. Quadratic elements should be able to reproduce the solution exactly, so the errors should reflect the machine precision.

# In[3]:


import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import (Expression, Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dot, dx, grad, inner

def error_infinity(u_h, u_ex):
    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    comm = u_h.function_space.mesh.comm
    u_ex_V = Function(u_h.function_space)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, u_h.function_space.element.interpolation_points)
        u_ex_V.interpolate(u_expr)
    else:
        u_ex_V.interpolate(u_ex)
    # Compute infinity norm, furst local to process, then gather the max
    # value over all processes
    error_max_local = np.max(np.abs(u_h.x.array-u_ex_V.x.array))
    error_max = comm.allreduce(error_max_local, op=MPI.MAX)
    return error_max


# In[4]:


error_infinity(uh, u_exact)


# **Questions and Comments**
# 
# 1.) The generalization of the Poisson problem for a uniform ellipse is a logical next step. The mesh-maker gmsh already has a pre-defined ellipsoidal mesh (this is in fact how the disc was created above). The boundary conditions will be provided via the exact analytic solution, which in 3 dimensions is given by Chandrasekhar by the formula 
# 
# $$\Phi(x) = \cases{\pi G \rho_0 [I(0) - \sum_{i=1}^3{A_i(0)x_i^2}],\quad x\quad\mathrm{inside} \\ 
# \pi G \rho_0 [I(\lambda) - \sum_{i=1}^3{A_i(\lambda)x_i^2}], \quad x\quad\mathrm{outside}} $$
# 
# where $a_i>0$ are the principle axes, and 
# $$I(u) = a_1 a_2 a_3 \int_{u}^\infty{\frac{du}{\Delta}},$$
# 
# $$ A_i(u) = a_1 a_2 a_3 \int_{u}^\infty{\frac{du}{\Delta(a_i^2 - u)}}$$
# 
# and $\Delta^2 = (a_1^2+u)(a_2^2 + u)(a_3^2 +u)$ with $\lambda$ is the largest root of 
# $$\sum_{i=1}^3{\frac{x_i^2}{a_i^2+\lambda}} = 1.$$
# 
# Note that the potential will no longer be constant on the surface of the ellipse.
# 
# 2.) The Poisson problem for the ellipse depends on the precise computation of the constants $\lambda, \, A_i(\lambda),\, I(\lambda)$. The wiki states that there is a Mathematica program that has been created for this purpose. Some test values for particular choices of $a_1,\,a_2,\,a_3$ are provided on the page https://tohline.education/SelfGravitatingFluids/index.php/ThreeDimensionalConfigurations/RiemannStype#TestPart1
