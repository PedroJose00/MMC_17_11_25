from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from pathlib import Path

from scipy.integrate import quad

# adiciono velocidad a la BMU
# +
import math
import ufl
import dolfinx 
import csv
import numpy as np
from dolfinx import la
from dolfinx.fem import (Expression, Function, FunctionSpace, dirichletbc,
                         form, functionspace, locate_dofs_topological, locate_dofs_geometrical)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.mesh import (CellType, GhostMode, create_box,
                          locate_entities_boundary)
from ufl import dx, grad, inner, tr, det , ln
from ufl import TensorElement, FiniteElement, VectorElement, FunctionSpace, as_tensor
import time
import pandas as pd
from scipy.optimize import newton

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Marcar el tiempo de inicio
start_time = time.time()
#time.sleep(1)  # Simula el cálculo pausando la ejecución por 1 segundo


from scipy.interpolate import griddata, LinearNDInterpolator

from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.mesh import meshtags_from_entities
from dolfinx.cpp.mesh import cell_entity_type
from dolfinx.io import distribute_entity_data
from dolfinx.graph import adjacencylist
from dolfinx.mesh import create_mesh
from dolfinx.cpp.mesh import to_type
from dolfinx.cpp.io import perm_gmsh
import numpy
import meshio
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx.io.gmshio import model_to_mesh
import numpy as np
import gmsh
import warnings
import meshio
warnings.filterwarnings("ignore")
dtype = PETSc.ScalarType  # type: ignore





dtype = PETSc.ScalarType  # type: ignore
# Suprimir advertencias (por ejemplo, de Gmsh cuando se trabaja con archivos STEP)
warnings.filterwarnings("ignore")





def build_nullspace(V: FunctionSpace):
    """Build PETSc nullspace for 3D elasticity"""

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(6)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    _basis = [x._cpp_object for x in basis]
    dolfinx.cpp.la.orthonormalize(_basis)
    assert dolfinx.cpp.la.is_orthonormal(_basis)

    basis_petsc = [PETSc.Vec().createWithArray(x[:bs * length0], bsize=3, comm=V.mesh.comm) for x in b]  # type: ignore
    # Suponiendo que tienes un espacio de funciones llamado 'V'
    #dof_per_node = V.dofmap.index_map_bs

# Imprimir la cantidad de DOF por nodo
#    print("Cantidad de DOF por nodo:", dof_per_node)
    return PETSc.NullSpace().create(vectors=basis_petsc)  # type: ignore

#----

atol=1e-3


# Inicializar Gmsh
gmsh.initialize()

# Crear un nuevo modelo
gmsh.model.add("modelo_3d")

# Cargar archivo STEP
archivo_step = "001.step"  # Reemplaza con la ruta a tu archivo STEP
gmsh.merge(archivo_step)

# Sincronizar para asegurar que la geometría se cargue en el modelo
gmsh.model.occ.synchronize()

volumes = gmsh.model.getEntities(dim=3)
bone_marker = 11
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], bone_marker)
gmsh.model.setPhysicalName(volumes[0][0], bone_marker, "bone_marker")
#______________________________

surfaces = gmsh.model.getEntities(dim=2)

# Asignar un identificador único a cada superficie
for i, surface in enumerate(surfaces, start=1):
    # Crear un grupo físico para la superficie actual
    group_id = gmsh.model.addPhysicalGroup(2, [surface[1]])
    # Opcional: asignar un nombre al grupo físico
    gmsh.model.setPhysicalName(2, group_id, f"Surface_{i}")



# ---------------------------------------------------------------------------------------------
# Definir el tamaño de los elementos de la malla
# Esto puede variar dependiendo de la geometría y los requerimientos de la malla
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 1.5)
# ---------------------------------------------------------------------------------------------
# Generar la malla de volumen (tetraedros)
gmsh.model.mesh.generate(3)



# Guardar la malla en un archivo
#gmsh.write("bone_mesh_1.5.msh")   ---->>>>>>>
 

# Obtener los tipos de elementos y sus tags
element_types, element_tags, node_tags = gmsh.model.mesh.getElements()

# # Iterar sobre los tipos de elementos y sus tags
# for elem_type, tags in zip(element_types, element_tags):
#     for tag in tags:
#         print(f"Elemento del tipo {elem_type}, ID: {tag}")

# gmsh.write("mesh3D.msh")
x = gmshio.extract_geometry(gmsh.model)
topologies = gmshio.extract_topology_and_markers(gmsh.model)

# Asumiendo que estás trabajando con una malla 2D con elementos triangulares
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
element_types, element_tags, node_tags_elem = gmsh.model.mesh.getElements()


# mesh = meshio.read("malla_salida10000.msh")

#     # Información de la malla
# num_nodes = len(mesh.points)
# num_elements = sum(len(cell.data) for cell in mesh.cells)
# cell_types = set(cell.type for cell in mesh.cells)
# print(f"Número de Nodos: {num_nodes}")
# print(f"Número de Elementos: {num_elements}")
# print(f"Tipos de Elementos: {cell_types}")
#___________________________________________________________________________


# Get information about each cell type from the msh files
num_cell_types = len(topologies.keys())
cell_information = {}
cell_dimensions = numpy.zeros(num_cell_types, dtype=numpy.int32)
for i, element in enumerate(topologies.keys()):
    properties = gmsh.model.mesh.getElementProperties(element)
    name, dim, order, num_nodes, local_coords, _ = properties
    cell_information[i] = {"id": element, "dim": dim,
                           "num_nodes": num_nodes}
    cell_dimensions[i] = dim


# Sort elements by ascending dimension
perm_sort = numpy.argsort(cell_dimensions)

#___________________________________________________________________________

#___________________________________________________________________________

cell_id = cell_information[perm_sort[-1]]["id"]
cells = numpy.asarray(topologies[cell_id]["topology"], dtype=numpy.int64)
ufl_domain = gmshio.ufl_mesh(cell_id, 3)

# #___________________________________________________________________________

#___________________________________________________________________________


num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
cells = cells[:, gmsh_cell_perm]

#___________________________________________________________________________

mesh = create_mesh(MPI.COMM_SELF, cells, x, ufl_domain)
# Obtener las coordenadas de todos los vértices de la malla
coordinates = mesh.geometry.x
#filename = "mesh_coordinates.txt"

# # Escribir las coordenadas en el archivo
# with open(filename, 'w') as file:
#     for i, coord in enumerate(coordinates):
#         file.write(f"Vértice {i}: {coord}\n")

# print(f"Las coordenadas de la malla se han guardado en '{filename}'")
#___________________________________________________________________________

# Create MeshTags for cell data
cell_values = numpy.asarray(
    topologies[cell_id]["cell_data"], dtype=numpy.int32)
local_entities, local_values = distribute_entity_data(
    mesh, mesh.topology.dim, cells, cell_values)
mesh.topology.create_connectivity(mesh.topology.dim, 0)
adj = adjacencylist(local_entities)
ct = meshtags_from_entities(mesh, mesh.topology.dim, adj, local_values)
ct.name = "Cell tags"

# Create MeshTags for facets
# Permute facets from MSH to DOLFINx ordering
# FIXME: This does not work for prism meshes
facet_type = cell_entity_type(
    to_type(str(ufl_domain.ufl_cell())), mesh.topology.dim - 1, 0)
gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
num_facet_nodes = cell_information[perm_sort[-2]]["num_nodes"]
gmsh_facet_perm = perm_gmsh(facet_type, num_facet_nodes)
marked_facets = numpy.asarray(
    topologies[gmsh_facet_id]["topology"], dtype=numpy.int64)
facet_values = numpy.asarray(
    topologies[gmsh_facet_id]["cell_data"], dtype=numpy.int32)
marked_facets = marked_facets[:, gmsh_facet_perm]
local_entities, local_values = distribute_entity_data(
    mesh, mesh.topology.dim - 1, marked_facets, facet_values)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
adj = adjacencylist(local_entities)
ft = meshtags_from_entities(mesh, mesh.topology.dim - 1, adj, local_values)
ft.name = "Facet tags"

# Output DOLFINx meshes to file

# with XDMFFile(MPI.COMM_WORLD, "bone_out_1.5.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(ft, mesh.geometry)
#     xdmf.write_meshtags(ct, mesh.geometry)
gmsh.finalize()


# Espacio CG1 de 36 componentes (Voigt aplanado)

def _as_vec36(arr_6x6_or_36):
    A = np.asarray(arr_6x6_or_36)
    if A.ndim == 3 and A.shape[1:] == (6, 6):
        A = A.reshape(-1, 36)
    elif A.ndim == 2 and A.shape[1] == 36:
        pass
    else:
        raise ValueError(f"Esperaba (n,6,6) o (n,36), recibí {A.shape}")
    return A

def save_vec36_xdmf(label: str, A_local, outdir="./28_10_25", overwrite=True):
    """
    A_local: ndarray local con shape (n_dofs_local_CG1, 6, 6) o (n_dofs_local_CG1, 36)
             (mismo orden que V.tabulate_dof_coordinates()).
    """
    comm = mesh.comm
    A = _as_vec36(A_local)
    a_flat = A.reshape(-1).astype(float)

    f = fem.Function(W, name=label)
    nloc = f.x.array.size
    if a_flat.size != nloc:
        # Ajuste defensivo (cortar o rellenar con 0)
        b = np.zeros(nloc, dtype=float); b[:min(nloc, a_flat.size)] = a_flat[:min(nloc, a_flat.size)]
        a_flat = b
    f.x.array[:] = a_flat
    f.x.scatter_forward()

    outdir = Path(outdir).resolve()
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for suf in (".xdmf", ".h5", ".bp"):
                p = outdir / f"fem_function_{label}{suf}"
                try: p.unlink()
                except FileNotFoundError: pass
    comm.Barrier()

    xdmf_path = str(outdir / f"fem_function_{label}.xdmf")
    with XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        try:
            xdmf.write_function(f, 0.0)
        except TypeError:
            xdmf.write_function(f)
    if comm.rank == 0:
        print(f"✓ guardado {label} -> {xdmf_path}")

def save_6x6_as_36_scalars(label: str, A_local, outdir="./28_10_25", overwrite=True):
    """
    A_local: ndarray (n_dofs_local_CG1, 6, 6) – componentes por nodo (CG1).
    Escribe 36 funciones escalares CG1: {label}_00 ... {label}_55
    """
    from pathlib import Path
    comm = mesh.comm
    A = np.asarray(A_local)
    if A.ndim != 3 or A.shape[1:] != (6,6):
        raise ValueError(f"{label}: se espera (n,6,6), llegó {A.shape}")

    V1 = fem.FunctionSpace(mesh, ("CG",1))
    nloc = V1.dofmap.index_map.size_local
    if A.shape[0] != nloc:
        B = np.zeros((nloc, 6, 6), dtype=float)
        B[:min(nloc, A.shape[0]), :, :] = A[:min(nloc, A.shape[0]), :, :]
        A = B

    outdir = Path(outdir).resolve()
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for suf in (".xdmf", ".h5", ".bp"):
                p = outdir / f"{label}_CG1_components{suf}"
                try: p.unlink()
                except FileNotFoundError: pass
    comm.Barrier()

    xdmf_path = str(outdir / f"{label}_CG1_components.xdmf")
    with XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        for i in range(6):
            for j in range(6):
                fij = fem.Function(V1, name=f"{label}_{i}{j}")
                fij.x.array[:nloc] = A[:, i, j]
                fij.x.scatter_forward()
                xdmf.write_function(fij)
    if comm.rank == 0:
        print(f"✓ guardado {label} (36 escalares) -> {xdmf_path}")

def expand_to_36(mesh_values):
    """Expande un array (n_dofs,) → (n_dofs, 36) replicando el valor."""
    arr = np.asarray(mesh_values, dtype=float)
    return np.tile(arr[:, None], (1, 36))

# Valor calcuala para Junio 6 de 2025---||| Bone
HU_BM = 1121.54  # Valor específico para la matriz ósea sólida
# Definimos las densidades en g/cm^3
rho_m = 3.1 # Densidad del mineral (g/cm^3)
rho_o = 1.1  # Densidad de la materia orgánica (g/cm^3)
Vo = 3/7  # Volumen orgánico (valor de ejemplo en cm^3)
#media_ref = 

# Asumimos que el archivo se llama 'propiedades.csv' y ya está en la ubicación correcta
archivo_propiedades = './ziou_23_08_24_cleaned.csv'
archivo_HU = './HU_15_10_24_cleaned_no_nan_negatives.csv'

# Leer el archivo
df = pd.read_csv(archivo_propiedades)
df_hu = pd.read_csv(archivo_HU)

print(df[['E', 'DENS']].head())
print(df_hu[['HU']].head())


# Asumiendo que el archivo tiene columnas 'x', 'y', 'z', 'densidad', 'E', 'nu'
puntos = df[['X', 'Y', 'Z']].to_numpy()
puntos_HU = df_hu[['X', 'Y', 'Z']].to_numpy()

#------------------------------------------------------------------------------
###  --->>>> valor de la densidad aparente esta kg/mm^3
#------------------------------------------------------------------------------
# 1) MISMO ESCALAR POR CELDA → MISMO VALOR EN LAS 36 COMPONENTES
#    (bloques [v0×36, v1×36, ...])

df['DENS'] = df['DENS']*1e-9

df_densidades = df['DENS'].to_numpy()
#df_densidades = np.repeat(df_densidades_TMP, 36)   # tamaño = n_local * 36

df_hu = df_hu['HU'].to_numpy()
#df_hu = np.repeat(df_hu_TMP, 36)

#------------------------------------------------------------------------------
#BVTV= 2.05e-6/densidades  
#modulos_E = 84370*BVTV**2.58*0.70**2.74*1e12
### EL modulo de elasticidad esta N/mm^2
#------------------------------------------------------------------------------
df['E']=df['E']*1e6            
df_modulos_E = df['E'].to_numpy()
#------------------------------------------------------------------------------
# El valor del módulo de poisson son es adimensional
df_coeficientes_nu = df['NU'].to_numpy()


# Calcula BVTV para cada punto
bvtv_values = np.where(df_hu <= HU_BM, (1-((HU_BM - df_hu) / HU_BM)), 1)
#bvtv_values = np.where(df_hu <= HU_BM, (1-((HU_BM - df_hu) / HU_BM)), 1)
#------------------------------------------------------------------------------
# los valores por defecto inicialmente la mediana del valor
#------------------------------------------------------------------------------
valor_por_defecto = np.median(df_densidades)  # O np.mean(valores_densidad)
valor_por_defecto_e = np.median(df_modulos_E)   
valor_por_defecto_nu = np.median(df_coeficientes_nu)
valor_por_defecto_HU = np.median(df_hu)   
valor_por_defecto_bvtv = np.median(bvtv_values)
#------------------------------------------------------------------------------


# 1) CG1 con 36 componentes (Voigt aplanado 6x6)
# el36_cg1 = VectorElement("Lagrange", mesh.ufl_cell(), 1, dim=36)
# W = fem.FunctionSpace(mesh, el36_cg1)            # CG1 (continuo), 36 comps

W =fem.FunctionSpace(mesh,("CG",1)) 

WDG0 = fem.FunctionSpace(mesh, ("DG",0))

# --- VectorElement DG-0 para 36 componentes por celda (piecewise constant) ---
C_dim = 36  # 6x6 aplanado por filas
Ve = ufl.VectorElement("DG", mesh.ufl_cell(), 0, dim=C_dim)
WC  = fem.FunctionSpace(mesh, Ve)
tdim = mesh.topology.dim
num_cells_local = mesh.topology.index_map(tdim).size_local



# ====== 3) Construir Cfib_local (flatten 36) en orden de celdas locales ======
# # Para DG-0 vector con dim=36, el arreglo interno tiene longitud = num_cells_local * 36
# Cfib_local_flat = np.zeros(num_cells_local * C_dim, dtype=float)

# # Funcion donde guardaremos C_fib aplanado
# fem_function_Cfib_vec = fem.Function(WC, name="C_fib_voigt_6x6")

# Cfib_local: np.ndarray de shape (num_celdas_local, 36) o (N*36,)
# Si lo tienes (N,36), aplánalo:
# fem_function_Cfib_vec.x.array[:] = Cfib_local_flat.reshape(-1)
# fem_function_Cfib_vec.x.scatter_forward()


fem_function_densidad_0= fem.Function(W, name="initial_density")
fem_function_densidad_aparente = fem.Function(W,name="aparent_density")
fem_function_modulo_E_0 = fem.Function(W,name="Young_module")
fem_function_coeficientes_nu_0 = fem.Function(W,name="initial_poisson")
fem_function_alpha = fem.Function(W, name="alpha")
fem_function_BVTV = fem.Function(W,name="BVTV")
fem_function_array_error= fem.Function(W, name="Error") 
fem_function_densidad_material=fem.Function(W,name="material_density")
fem_function_dot_module = fem.Function(W,name= "dot_young")
fem_function_HU = fem.Function(W,name="HU")
#fem_function_cenizas = fem.Function(W,name="alpha")
fem_function_BMD = fem.Function(W,name="BMD")
fem_function_BMD_surface = fem.Function(W,name="BMD_surface")
fem_function_T_score = fem.Function(W,name="T_score")
fem_function_phi = fem.Function(W,name = "porosity")
fem_function_nu = fem.Function(W,name="poisson")
fem_function_E = fem.Function(W,name="E_hill")
fem_function_rho_app = fem.Function(W,name="rho_app")
fem_function_rho_mat = fem.Function(W, name="rho_mat")
fem_function_rho_ash_app = fem.Function(W,name="rho_ash_app")
fem_function_rho_ash_mat = fem.Function(W, name="rho_ash_mat")
# --- NUEVAS funciones: ORGÁNICO y AGUA (tejido/aparente) ---
fem_function_rho_org_mat = fem.Function(W, name="rho_org_mat")  # orgánico (tejido)
fem_function_rho_org_app = fem.Function(W, name="rho_org_app")  # orgánico (aparente)
fem_function_rho_wat_mat = fem.Function(W, name="rho_wat_mat")  # agua (tejido)
fem_function_rho_wat_app = fem.Function(W, name="rho_wat_app")  # agua (aparente)

# --- (Opcional) Fracciones tisulares para visualizar en ParaView ---
fem_function_nu_o = fem.Function(W, name="nu_o")  # fracción orgánica tisular
fem_function_nu_w = fem.Function(W, name="nu_w")  # fracción de agua tisular
fem_function_nu_m = fem.Function(W, name = "nu_m")
# Funciones para calcular fracciones de volumen de colageno - HA y agua
# fem_function_bvtv_wetcol = fem.Function(WDG0, name="bvtv_wetcol") 
# fem_function_bvtv_HA_in = fem.Function(WDG0, name="bvtv_HA_in") 
# fem_function_bvtv_water = fem.Function(WDG0, name="bvtv_water") 


fem_function_bvtv_wetcol = fem.Function(WDG0, name="bvtv_wetcol")  # fracción orgánica tisular
fem_function_bvtv_HA_in= fem.Function(WDG0, name="bvtv_HA_in")  # fracción de agua tisular
fem_function_bvtv_ic = fem.Function(WDG0, name = "bvtv_ic")

fem_function_bvtv_fib = fem.Function(WDG0, name="bvtv_fib")  # fracción orgánica tisular
fem_function_bvtv_bvtv_ef= fem.Function(WDG0, name="bvtv_ef")  # fracción de agua tisular
#-----------------------------------------------------------------------------
fem_function_E1 = fem.Function(W, name="E1")  #Voight and Reuss
fem_function_E2 = fem.Function(W, name="E2")  # fracción de agua tisular
fem_function_E3 = fem.Function(W, name = "E3")
fem_function_nu12 = fem.Function(W, name = "nu12")
fem_function_nu13 = fem.Function(W, name = "nu13")
fem_function_nu23 = fem.Function(W, name = "nu23")

#-----------------------------------------------------------------------------


fem_function_pct_mineral_vol  = fem.Function(W, name="pct_mineral_vol")
fem_function_pct_mineral_masa = fem.Function(W, name="pct_mineral_masa")
fem_function_pct_org_vol  = fem.Function(W, name="pct_org_vol")
fem_function_pct_org_masa = fem.Function(W, name="pct_org_masa")
fem_function_phi = fem.Function(W, name="phi")
fem_function_BSBV = fem.Function(W, name="BSBV")
fem_function_BSTV = fem.Function(W,name="BSTV")
fem_function_TbTh = fem.Function(W,name="TbTh")
fem_function_TbSp = fem.Function(W,name="TbSp")
fem_function_TbN = fem.Function(W,name="TbN")


# Obtener las coordenadas de los centros de los elementos o nodos de la malla Fenicsx
mesh_coordinates = W.tabulate_dof_coordinates()  # Asumiendo que W es tu espacio de funciones DG0
mesh_coordinates_DG0 = WDG0.tabulate_dof_coordinates()
# Extracción de valores de la función FEniCS/DOLFINx (Este es un paso conceptual, ajusta según sea necesario)
#valores_densidad = [fem_function_densidad_0.eval(punto) for punto in puntos]  # Ajusta esta línea según sea necesario



# Crear el interpolador
linear_interp_densidad_0 = LinearNDInterpolator( puntos , df_densidades)
linear_interp_hu = LinearNDInterpolator(puntos_HU, df_hu)
linear_interp_bvtv = LinearNDInterpolator(puntos_HU, bvtv_values)
#interpolador_nu = LinearNDInterpolator(puntos, coeficientes_nu)


# Usar el interpolador para obtener valores de densidad en las coordenadas de la malla
mesh_value_densidad_0 = linear_interp_densidad_0(mesh_coordinates)
#mesh_value_densidad_0  = np.repeat(mesh_value_densidad_0_TMP, 36)   # tamaño = n_local * 36

#mesh_value_modulo_E_0 = linear_interp_modulo_E_0(mesh_coordinates)
mesh_value_hu = linear_interp_hu(mesh_coordinates)
#mesh_value_hu = np.repeat(mesh_value_hu_TMP, 36)   # tamaño = n_local * 36

#valor_por_defecto_bvtv_TMP = np.median(bvtv_values)

mesh_value_bvtv = linear_interp_bvtv(mesh_coordinates)
#mesh_value_bvtv = np.repeat(mesh_value_bvtv_TMP, 36)   # tamaño = n_local * 36



#valor_por_defecto_bvtv= np.median(bvtv_values)
#valor_por_defecto_bvtv  = np.repeat(mesh_value_bvtv , 36)   # tamaño = n_local * 36


# Manejar posibles NaNs después de la interpolación
mesh_value_densidad_0  = np.nan_to_num(mesh_value_densidad_0, nan=valor_por_defecto)
#mesh_value_modulo_E_0  = np.nan_to_num(mesh_value_modulo_E_0 , nan=valor_por_defecto_e)
mesh_value_hu = np.nan_to_num(mesh_value_hu,nan=valor_por_defecto_HU)
mesh_value_bvtv = np.nan_to_num(mesh_value_bvtv, nan=valor_por_defecto_bvtv)

#valores_interpolados_nu = np.nan_to_num(valores_interpolados, nan=valor_por_defecto_nu)

# Asumiendo que densidad_h es tu función Fenicsx en el espacio DG0
# Aca se extraeen los valores de densidad del dominio pra poder determinar 
# el valor de la porosidad tejido se calcula con el dato densidad
fem_function_densidad_0.vector.setArray(mesh_value_densidad_0)
#fem_function_modulo_E_0.vector.setArray(mesh_value_modulo_E_0)
fem_function_HU.vector.setArray(mesh_value_hu)
fem_function_BVTV.vector.setArray(mesh_value_bvtv)

# mesh_value_densidad_36 = expand_to_36(mesh_value_densidad_0)
# mesh_value_bvtv_36 = expand_to_36(mesh_value_bvtv)
# mesh_value_hu_36 = expand_to_36(mesh_value_hu)

# fem_function_densidad_0.x.array[:] = mesh_value_densidad_36.reshape(-1)
# fem_function_BVTV.x.array[:] = mesh_value_bvtv_36.reshape(-1)
# fem_function_HU.x.array[:] = mesh_value_hu_36.reshape(-1)


# save_vec36_xdmf("densidad_6x6", mesh_value_densidad_36)
# save_vec36_xdmf("BVTV_6x6", mesh_value_bvtv_36)
# save_vec36_xdmf("HU_6x6", mesh_value_hu_36)


fem_function_rho_app.vector.setArray(mesh_value_bvtv)
fem_function_rho_mat.vector.setArray(mesh_value_bvtv)
fem_function_rho_ash_app.vector.setArray(mesh_value_bvtv)
fem_function_rho_ash_mat.vector.setArray(mesh_value_bvtv)

fem_function_rho_org_mat.vector.setArray(mesh_value_bvtv) # orgánico (tejido)
fem_function_rho_org_app.vector.setArray(mesh_value_bvtv)
fem_function_rho_wat_mat.vector.setArray(mesh_value_bvtv)
fem_function_rho_wat_app.vector.setArray(mesh_value_bvtv)
# --- (Opcional) Fracciones tisulares para visualizar en ParaView ---
fem_function_nu_o.vector.setArray(mesh_value_bvtv)
fem_function_nu_w.vector.setArray(mesh_value_bvtv)
fem_function_nu_m.vector.setArray(mesh_value_bvtv)

fem_function_pct_org_vol.vector.setArray(mesh_value_bvtv)
fem_function_pct_org_masa.vector.setArray(mesh_value_bvtv)
fem_function_phi.vector.setArray(mesh_value_bvtv)
fem_function_BSBV.vector.setArray(mesh_value_bvtv)
fem_function_BSTV.vector.setArray(mesh_value_bvtv)
fem_function_TbTh.vector.setArray(mesh_value_bvtv)
fem_function_TbN.vector.setArray(mesh_value_bvtv)
fem_function_TbSp.vector.setArray(mesh_value_bvtv)
#--------------------------------------------------------
fem_function_E1.vector.setArray(mesh_value_bvtv)
fem_function_E2.vector.setArray(mesh_value_bvtv)
fem_function_E3.vector.setArray(mesh_value_bvtv)
fem_function_nu12.vector.setArray(mesh_value_bvtv)
fem_function_nu13.vector.setArray(mesh_value_bvtv)
fem_function_nu23.vector.setArray(mesh_value_bvtv)

#--------------------------------------------------------

array_modulo_E_0 = fem_function_modulo_E_0.x.array[:]
array_densidad_0= fem_function_densidad_0.x.array[:]
array_hu = fem_function_HU.x.array[:]
array_densidad = fem_function_densidad_0.x.array[:]
array_BVTV = fem_function_BVTV.x.array[:]
array_phi = fem_function_phi.x.array[:]
array_nu = fem_function_nu.x.array[:]
array_E = fem_function_E.x.array[:]
array_rho_app = fem_function_rho_app.x.array[:] 
array_rho_mat = fem_function_rho_mat.x.array[:]
array_rho_ash_app = fem_function_rho_ash_app.x.array[:] 
array_rho_ash_mat = fem_function_rho_ash_mat.x.array[:]
a_rho_org_mat= fem_function_rho_org_mat.x.array[:]
a_rho_org_app = fem_function_rho_org_app.x.array[:]
a_rho_wat_mat =fem_function_rho_wat_mat.x.array[:]
a_rho_wat_app =  fem_function_rho_wat_app.x.array[:]
nu_o =fem_function_nu_o.x.array[:]
nu_w = fem_function_nu_w.x.array[:]
nu_m = fem_function_nu_m.x.array[:]
mineral_vol_pct = fem_function_pct_mineral_vol.x.array[:]
mineral_mass_pct = fem_function_pct_mineral_masa.x.array[:]
pct_org_vol = fem_function_pct_org_vol.x.array[:]
pct_org_masa = fem_function_pct_org_masa.x.array[:]

array_BSBV = fem_function_BSBV.x.array[:] 
array_BSTV = fem_function_BSTV.x.array[:]
array_TbTh = fem_function_TbTh.x.array[:]
array_TbN =  fem_function_TbN.x.array[:]
array_TbSp =  fem_function_TbSp.x.array[:]
bvtv_wetcol = fem_function_bvtv_wetcol.x.array[:]
#--------------------------------------------------------

E1_arr = fem_function_E1.x.array[:]
E2_arr = fem_function_E2.x.array[:]
E3_arr = fem_function_E3.x.array[:]
nu12_arr = fem_function_nu12.x.array[:]
nu13_arr = fem_function_nu13.x.array[:]
nu23_arr = fem_function_nu23.x.array[:]





#--------------------------------------------------------


# 1) Espacio DG0 en celdas (usa WE si quieres mantener tu nombre)
WE = fem.FunctionSpace(mesh, ("DG", 0))

# 2) Número de celdas OWNED (locales) en este rank
n_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local

# 3) Helper para ajustar tamaño al número de celdas locales
def _fit_to_local(a: np.ndarray, n_local: int) -> np.ndarray:
    out = np.zeros(n_local, dtype=np.float64)
    if a is None:
        return out
    a = np.asarray(a, dtype=np.float64)
    n = min(a.size, n_local)
    out[:n] = np.nan_to_num(a[:n], nan=0.0, posinf=0.0, neginf=0.0)
    return out

# --- Asegura tamaños coherentes para cada campo direccional ---
E1_local   = _fit_to_local(E1_arr,   n_cells_local)
E2_local   = _fit_to_local(E2_arr,   n_cells_local)
E3_local   = _fit_to_local(E3_arr,   n_cells_local)
nu12_local = _fit_to_local(nu12_arr, n_cells_local)
nu13_local = _fit_to_local(nu13_arr, n_cells_local)
nu23_local = _fit_to_local(nu23_arr, n_cells_local)

# --- Crea funciones DG0 (un dof por celda) ---
f_E1   = fem.Function(WE, name="E1")
f_E2   = fem.Function(WE, name="E2")
f_E3   = fem.Function(WE, name="E3")
f_nu12 = fem.Function(WE, name="nu12")
f_nu13 = fem.Function(WE, name="nu13")
f_nu23 = fem.Function(WE, name="nu23")

# --- Asigna arrays (para DG0, orden de dofs == orden de celdas locales) ---
f_E1.x.array[:]   = E1_local
f_E2.x.array[:]   = E2_local
f_E3.x.array[:]   = E3_local
f_nu12.x.array[:] = nu12_local
f_nu13.x.array[:] = nu13_local
f_nu23.x.array[:] = nu23_local




f_Eiso  = fem.Function(WE, name="E_iso")
f_nuiso = fem.Function(WE, name="nu_iso")
f_KH    = fem.Function(WE, name="K_H")
f_GH    = fem.Function(WE, name="G_H")

# ---------- asignar ----------
f_E1.x.array[:]   = E1_local
f_E2.x.array[:]   = E2_local
f_E3.x.array[:]   = E3_local
# f_nu12.x.array[:] = np.clip(nu12_local, -0.49, 0.49)
# f_nu13.x.array[:] = np.clip(nu13_local, -0.49, 0.49)
# f_nu23.x.array[:] = np.clip(nu23_local, -0.49, 0.49)




# ---------------------------------------------------------------------------
# Si prefieres mantener tus variables existentes (ojo con lo que significan):
# fem_function_E2, fem_function_nu2, fem_function_K2, fem_function_G2
# Asegúrate de que W sea DG0 y asigna:
# fem_function_E2.x.array[:]  = _fit_to_local(E2_arr,   n_cells_local)
# fem_function_nu2.x.array[:] = _fit_to_local(nu12_arr, n_cells_local)  # si 'nu2' = nu12
# fem_function_K2.x.array[:]  = _fit_to_local(KH_arr,   n_cells_local)   # si tienes K direccional o equivalente
# fem_function_G2.x.array[:]  = _fit_to_local(GH_arr,   n_cells_local)
# ---------------------------------------------------------------------------

# # --- (Opcional) Exporta a ParaView ---
# with XDMFFile(MPI.COMM_WORLD, "direccionales_ultra.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(f_E1,   0.0)
#     xdmf.write_function(f_E2,   0.0)
#     xdmf.write_function(f_E3,   0.0)
#     xdmf.write_function(f_nu12, 0.0)
#     xdmf.write_function(f_nu13, 0.0)
#     xdmf.write_function(f_nu23, 0.0)



##----------------------------------------
# el valor de la densidad para calcular el modulo debe estar en gr/cm3
array_densidad_gr_cm3 = array_densidad*1e6
fem_function_densidad_aparente.x.array[:] = array_densidad_gr_cm3 
# Calcula el valor promedio de array_densidad
promedio_densidad = np.mean(array_densidad_gr_cm3)

print("El valor promedio de la densidad es:", promedio_densidad )
print("00__Hay NaN en expression_array_BVTV:", np.isnan(array_BVTV).any())
fem_function_BVTV.x.array[:]= array_BVTV


array_material = array_densidad_gr_cm3 /array_BVTV
fem_function_densidad_material.x.array[:] = array_material


# ===================== SOLO CT: cálculo + export por campo =====================


# ---- Constantes de fases (g/cm^3) y orgánico tisular ----
RHO_MIN, RHO_ORG, RHO_WAT = 3.1, 1.1, 1.00
NU_O = 3.0/7.0
EPS  = 1e-6
EPS  = 1e-6

# ---------- Utils ----------
def print_stats(nombre: str, arr: np.ndarray):
    a = np.asarray(arr, float)
    print(f"{nombre:>16s}  min={np.nanmin(a):.4g}  mean={np.nanmean(a):.4g}  max={np.nanmax(a):.4g}")

outdir = Path("./05_10_25"); outdir.mkdir(parents=True, exist_ok=True)



# ---------- utilidades ----------
def _nz(x, eps=1e-12):
    x = float(x)
    return x if abs(x) > eps else (eps if x >= 0 else -eps)

def _nanfix(a):
    return np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def _fit_local(a, n_local):
    out = np.zeros(n_local, dtype=float)
    if a is None: return out
    a = _nanfix(a)
    n = min(a.size, n_local)
    out[:n] = a[:n]
    return out

# ---------- ct_only robusta ----------
def ct_only(array_densidad_gr_cm3: np.ndarray,
            array_BVTV: np.ndarray,
            *,
            # constantes con defaults (cámbialas a tus calibraciones)
            EPS: float = 1e-6,
            NU_O: float = 0.30,    # fracción orgánica tisular (vol)
            RHO_MIN: float = 3.00, # g/cm^3 (mineral/HA tisular efectiva)
            RHO_ORG: float = 1.30, # g/cm^3 (orgánico tisular)
            RHO_WAT: float = 1.00, # g/cm^3 (agua)
            fit_to_local: bool = False,
            mesh=None):
    """
    Entradas:
      - array_densidad_gr_cm3: ρ_app (g/cm^3) desde CT
      - array_BVTV           : BV/TV [0..1] desde CT
    Salidas (dict) con todas las claves que usas aguas abajo.
    """

    rho_app = _nanfix(array_densidad_gr_cm3)
    BVTV    = np.clip(_nanfix(array_BVTV), EPS, 1.0)  # evita 0

    # Opcional: ajustar a tamaño local DG0 para evitar mismatch al escribir
    if fit_to_local:
        if mesh is None:
            raise ValueError("ct_only(fit_to_local=True) requiere 'mesh'.")
        n_local = mesh.topology.index_map(mesh.topology.dim).size_local
        rho_app = _fit_local(rho_app, n_local)
        BVTV    = _fit_local(BVTV,    n_local)

    # Densidad tisular (material): rho_mat = rho_app / BVTV
    rho_mat = rho_app / np.clip(BVTV, _nz(EPS), 1.0)

    # Fracción orgánica tisular (constante)
    nu_o = np.full_like(rho_mat, float(NU_O))

    # Resolver nu_m de mezcla lineal de densidades:
    # rho_mat = nu_m*RHO_MIN + nu_o*RHO_ORG + nu_w*RHO_WAT,   nu_w = 1 - nu_m - nu_o
    # => rho_mat = nu_m*(RHO_MIN - RHO_WAT) + nu_o*(RHO_ORG - RHO_WAT) + RHO_WAT
    num = rho_mat - RHO_WAT - nu_o*(RHO_ORG - RHO_WAT)
    den = _nz(RHO_MIN - RHO_WAT)  # evita 0
    nu_m = num / den
    # recortes físicos
    nu_m = np.clip(nu_m, 0.0, 1.0 - nu_o)
    nu_w = 1.0 - nu_m - nu_o
    nu_w = np.clip(nu_w, 0.0, 1.0)

    # Densidades tisulares parciales (g/cm^3) por fase
    rho_ash_mat = RHO_MIN * nu_m
    rho_org_mat = RHO_ORG * nu_o
    rho_wat_mat = RHO_WAT * nu_w

    # Proyección aparente (multiplicar por BV/TV)
    rho_ash_app = rho_ash_mat * BVTV
    rho_org_app = rho_org_mat * BVTV
    rho_wat_app = rho_wat_mat * BVTV

    # Alpha: cenizas / masa seca
    denom_seco = (RHO_MIN * nu_m + RHO_ORG * nu_o)
    alpha = np.divide(RHO_MIN * nu_m, np.clip(denom_seco, _nz(EPS), None))
    alpha = np.clip(alpha, 0.0, 1.0)

    # Fracciones por masa (tisular)
    rho_mat_nz = np.clip(rho_mat, _nz(EPS), None)
    water_mass_frac   = (nu_w * RHO_WAT) / rho_mat_nz
    org_mass_frac     = (nu_o * RHO_ORG) / rho_mat_nz
    mineral_mass_frac = (nu_m * RHO_MIN) / rho_mat_nz

    # Por volumen (tisular)
    water_vol_frac   = nu_w
    org_vol_frac     = nu_o
    mineral_vol_frac = nu_m

    # % (0–100)
    water_vol_pct     = 100.0 * water_vol_frac
    water_mass_pct    = 100.0 * water_mass_frac
    org_vol_pct       = 100.0 * org_vol_frac
    org_mass_pct      = 100.0 * org_mass_frac
    mineral_vol_pct   = 100.0 * mineral_vol_frac
    mineral_mass_pct  = 100.0 * mineral_mass_frac

    # Chequeos de consistencia
    resid_vol_sum = np.abs((nu_m + nu_o + nu_w) - 1.0)

    rho_mat_recomp = rho_ash_mat + rho_org_mat + rho_wat_mat
    resid_rho_mat_recomp = np.abs(rho_mat_recomp - rho_mat)

    rho_app_recomp = rho_ash_app + rho_org_app + rho_wat_app
    resid_rho_app_recomp = np.abs(rho_app_recomp - rho_app)

    return dict(
        BVTV=BVTV, rho_app=rho_app, rho_mat=rho_mat,
        nu_m=nu_m, nu_o=nu_o, nu_w=nu_w,
        water_vol_frac=water_vol_frac, water_mass_frac=water_mass_frac,
        water_vol_pct=water_vol_pct,  water_mass_pct=water_mass_pct,
        org_vol_frac=org_vol_frac,   org_mass_frac=org_mass_frac,
        org_vol_pct=org_vol_pct,     org_mass_pct=org_mass_pct,
        rho_ash_mat=rho_ash_mat, rho_ash_app=rho_ash_app,
        rho_org_mat=rho_org_mat, rho_org_app=rho_org_app,
        rho_wat_mat=rho_wat_mat, rho_wat_app=rho_wat_app,
        alpha=alpha,
        resid_vol_sum=resid_vol_sum,
        mineral_vol_frac=mineral_vol_frac,
        mineral_mass_frac=mineral_mass_frac,
        mineral_vol_pct=mineral_vol_pct,
        mineral_mass_pct=mineral_mass_pct,
        resid_rho_mat_recomp=resid_rho_mat_recomp,
        resid_rho_app_recomp=resid_rho_app_recomp
    )

# ---- 1) Leer BV/TV de tu Function (DG0) ----
array_BVTV = fem_function_BVTV.x.array[:].astype(float)  # BV/TV ∈ [0,1]
BVTV = np.clip(array_BVTV, 0., 1)

# ---- 2) Porosidad ----
array_phi = 1-BVTV
phi = np.clip(array_phi, 0.0, 1)

# modelo Phyrie 23.481301
# modelo Richard parabolic
array_BSTV = 12.352*phi*(1-phi) 
# ---- 3) Modelo de Martin: Sv(φ) = BS/BV (mm^-1)  ***NO dividir por TV***
#p2 = phi*phi;p3= p2*phi; p4=p3*phi; p5=p4*phi; p6=p5*phi ; p7=p6*phi
  # φ = 1 - BV/TV
#array_BSBV = -0.9253*p5 + 3.07113*p4 -3.30755*p3  -1.49259*p2 + 6.65436*phi  -0.00104997
# Grado 2
#array_BSBV= -18.293*p2 + 20.5023*phi -0.546944
# Grado 3
#array_BSBV=-16.8446*p3 + 10.3593*p2 + 6.86561*phi + 0.90531
#phi = np.clip(array_phi, 0.0, 1)
# Grado 6
#array_BSBV =  33.2763*p6 + -212.032*p5 + 332.117*p4 + -187.427*p3 + 15.1424*p2 + 19.4738*phi + 0.000375085
# Grado 7
#array_BSBV = -1698.31+p7 + 6650.03*p6  -10607*p5 + 8714.36*p4 -3820.35*p3 + 809.819*p2  -48.1664*phi + 0.0110074
# ---- 3) Modelo de Martin: Sv(φ) = BS/BV (mm^-1)  ***NO dividir por TV***
#p2 = phi*phi; p3 = p2*phi; p4 = p3*phi; p5 = p4*phi
#array_BSBV= 11.744*p5 + -24.8288*p4 + 12.7578*p3+ -8.10107*p2 + 8.73836*phi + 0.364763
#S_v(\rho_{app}) = 36.31 \, \rho_{app} - 23.53 \, \rho_{app}^2 + 6.53 \, \rho_{app}^3 - 0.65 \, \rho_{app}^4
#array_BSTV = -117.812*p5 + 233.152*p4 -139.207*p3 + 3.86752*p2 + 20.553*phi + 0.00395944
# S_v = 28.76 p^5 − 101.4 p^4 + 133.96 p^3 − 93.94 p^2 + 32.26 
#array_BSBV = (28.76*p5 - 101.4*p4 + 133.96*p3 - 93.94*p2 + 32.26*phi)
#array_BSBV = (0.03226*phi - 0.09394*p2 + 0.13396*p3 - 0.10104*p4 + 0.02876*p5) *1000# modelo  de Martin used By Carter & Sons
#array_BSBV = np.clip(np.nan_to_num(array_BSBV, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)  # BS/BV ≥ 0

# ---- 4) BS/TV = (BS/BV) * (BV/TV) ----
#array_BSTV = array_BSBV*BVTV

# ---- 5) Morfometría (modelo plate-like: k=2). Para rod-like usa k=4. ----

array_BSBV = array_BSTV/BVTV
array_TbTh = BVTV / (array_BSTV + EPS)              # mm
array_TbN  = BVTV / (array_TbTh + EPS)            # mm^-1
array_TbSp = (phi/ (array_TbN + EPS))   # mm

# ===================== IMPRESIÓN =====================
print_stats("BV/TV (-)", BVTV)
print_stats("Porosidad φ", array_phi)
# print_stats("ρ_app (g/cm³)", ct_fields["rho_app"])
# print_stats("ρ_mat (g/cm³)", ct_fields["rho_mat"])
# print_stats("% agua (vol)", ct_fields["water_vol_pct"])
# print_stats("% mineral (vol)", ct_fields["mineral_vol_pct"])
# print_stats("% orgánico (vol)", ct_fields["org_vol_pct"])
# print_stats("BS/BV (mm⁻¹)", array_Sv_BSBV)
print_stats("BS/TV (mm⁻¹)", array_BSTV)
print_stats("Tb.Th (mm)", array_TbTh)
print_stats("Tb.N (mm⁻¹)", array_TbN)
print_stats("Tb.Sp (mm)", array_TbSp)



# ---- 2) Impresión rápida (por clave) ----
def print_stats(nombre: str, arr: np.ndarray):
    arr = np.asarray(arr, float)
    print(f"{nombre:>16s}  min={np.nanmin(arr):.4g}  mean={np.nanmean(arr):.4g}  max={np.nanmax(arr):.4g}")

# ---- 3) Guardar XDMF (un campo a la vez) ----
outdir = Path("./28_10_25"); outdir.mkdir(parents=True, exist_ok=True)

comm = mesh.comm  # usa el comunicador del mesh

def _fit_to_local(a: np.ndarray, n_local: int) -> np.ndarray:
    out = np.zeros(n_local, dtype=float)
    if a is None:
        return out
    a = np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0, neginf=0.0)
    n = min(a.size, n_local)
    out[:n] = a[:n]
    return out

def save_xdmf_scalar(label: str, func, arr, mesh, t: float = 0.0, overwrite=True, outdir="./28_10_25"):
    """
    Guarda un escalar DG0. Si XDMF/HDF5 paralelo falla, cae a ADIOS2 (.bp).
    """
    outdir = Path(outdir).resolve()
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Adaptar arreglo al tamaño local del Function
    a_local = _fit_to_local(arr, func.x.array.size)
    if label == "BVTV":
        a_local = np.clip(a_local, 0.0, 1.0)
    func.x.array[:] = a_local
    try:
        func.name = label
    except Exception:
        pass

    # Nombres de salida
    xdmf_path = (outdir / f"fem_function_{label}.xdmf")
    h5_path   = (outdir / f"fem_function_{label}.h5")   # backend usado por XDMF
    bp_path   = (outdir / f"fem_function_{label}.bp")   # fallback ADIOS2

    # Borrar previos de forma coordinada
    if comm.rank == 0 and overwrite:
        for p in (xdmf_path, h5_path, bp_path):
            try: p.unlink()
            except FileNotFoundError: pass
    comm.Barrier()

    # ====== Intento 1: XDMF paralelo (HDF5 paralelo requerido) ======
    try:
        with XDMFFile(comm, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(mesh)
            try:
                xdmf.write_function(func, t)
            except TypeError:
                xdmf.write_function(func)
        if comm.rank == 0:
            print(f"✓ guardado {label:>12s} -> {xdmf_path.name}")
        comm.Barrier()
        return
    except RuntimeError as e:
        # Solo rank 0 avisa una vez
        if comm.rank == 0:
            print(f"[warn] XDMF/HDF5 paralelo falló para '{label}': {e}")
            print("       Intentando fallback ADIOS2 (.bp) con VTXWriter…")
        comm.Barrier()


# ===================== EJEMPLO DE USO (solo lo que quieras) =====================


# === SALIDA de SOLO CT: usa un nombre distinto ===
ct_fields = ct_only(array_densidad_gr_cm3=array_densidad_gr_cm3,
                    array_BVTV=array_BVTV)
# 1) Calcula todo desde CT:
print_stats("ν_m (-)",          ct_fields["nu_m"])
print_stats("ν_o (-)",          ct_fields["nu_o"])
print_stats("ν_w (-)",          ct_fields["nu_w"])
print_stats("w_o (-)", ct_fields["org_mass_frac"])     # fracción másica orgánica
print_stats("%agua_vol",        ct_fields["water_vol_pct"])
print_stats("%agua_masa",       ct_fields["water_mass_pct"])
print_stats("%org_vol",         ct_fields["org_vol_pct"])
print_stats("%org_masa",        ct_fields["org_mass_pct"])
print_stats("resid_vol_sum",    ct_fields["resid_vol_sum"])
print_stats("resid_rho_mat",    ct_fields["resid_rho_mat_recomp"])
print_stats("resid_rho_app",    ct_fields["resid_rho_app_recomp"])
print_stats("mean %mineral_vol :", ct_fields["mineral_vol_pct"])
print_stats("mean %mineral_masa:", ct_fields["mineral_mass_pct"])
print_stats("%org_vol",  ct_fields["org_vol_pct"])
print_stats("%org_masa", ct_fields["org_mass_pct"])

# 3) Guarda UNO por vez (tú controlas qué y cuándo):
#    (descomenta los que quieras escribir)
save_xdmf_scalar("BVTV",          fem_function_BVTV,              ct_fields["BVTV"],          mesh)
save_xdmf_scalar("aparente",      fem_function_densidad_aparente, ct_fields["rho_app"],       mesh)
save_xdmf_scalar("material",      fem_function_densidad_material, ct_fields["rho_mat"],       mesh)
save_xdmf_scalar("alpha",         fem_function_alpha,             ct_fields["alpha"],         mesh)
save_xdmf_scalar("rho_ash_mat",   fem_function_rho_ash_mat,       ct_fields["rho_ash_mat"],   mesh)
save_xdmf_scalar("rho_ash_app",   fem_function_rho_ash_app,       ct_fields["rho_ash_app"],   mesh)


save_xdmf_scalar("rho_org_mat", fem_function_rho_org_mat, ct_fields["rho_org_mat"], mesh)
save_xdmf_scalar("rho_org_app", fem_function_rho_org_app, ct_fields["rho_org_app"], mesh)
save_xdmf_scalar("rho_wat_mat", fem_function_rho_wat_mat, ct_fields["rho_wat_mat"], mesh)
save_xdmf_scalar("rho_wat_app", fem_function_rho_wat_app, ct_fields["rho_wat_app"], mesh)
save_xdmf_scalar("nu_m",        fem_function_nu_m,        ct_fields["nu_m"],        mesh)
save_xdmf_scalar("nu_o",        fem_function_nu_o,        ct_fields["nu_o"],        mesh)
save_xdmf_scalar("nu_w",        fem_function_nu_w,        ct_fields["nu_w"],        mesh)
save_xdmf_scalar("alpha", fem_function_alpha, ct_fields["alpha"], mesh)
save_xdmf_scalar("pct_mineral_vol",  fem_function_pct_mineral_vol,  ct_fields["mineral_vol_pct"],  mesh)
save_xdmf_scalar("pct_mineral_masa", fem_function_pct_mineral_masa, ct_fields["mineral_mass_pct"], mesh)
save_xdmf_scalar("pct_org_vol",  fem_function_pct_org_vol,  ct_fields["org_vol_pct"],  mesh)
save_xdmf_scalar("pct_org_masa", fem_function_pct_org_masa, ct_fields["org_mass_pct"], mesh)

# --- Estereología / Martin ---
save_xdmf_scalar("phi",      fem_function_phi,  array_phi,   mesh)
save_xdmf_scalar("BSBV",     fem_function_BSBV, array_BSBV,  mesh)   # BS/BV (mm^-1)
save_xdmf_scalar("BSTV",     fem_function_BSTV, array_BSTV,  mesh)   # BS/TV (mm^-1)
save_xdmf_scalar("TbTh_mm",  fem_function_TbTh, array_TbTh,  mesh)   # Tb.Th (mm)
save_xdmf_scalar("TbN_mm1",  fem_function_TbN,  array_TbN,   mesh)   # Tb.N (mm^-1)
save_xdmf_scalar("TbSp_mm",  fem_function_TbSp, array_TbSp,  mesh)   # Tb.Sp (mm)





# ============
# Fracciones V
# ============
# Etapa 1 (intermolecular): bvtv_im = poro acuoso entre moléculas de colágeno
bvtv_im = 0.43     # agua intermolecular
f_im    = bvtv_im
f_col   = 1.0 - f_im

# Etapa 2 (fibrilla): wet collagen + HA
bvtv_wetcol = nu_o   # fracción en la fibrilla ocupada por wet-collagen
bvtv_HA_in  = nu_m

# Etapa 3 (espuma extrafibrilar): HA + agua intercristalina
bvtv_HA_ef = nu_m
bvtv_ic    = nu_w

# Etapa 4 (ultra): extrafibrilar + fibrilla
bvtv_fib  = nu_o + nu_m
bvtv_ef   = nu_m + nu_w

# Etapa 5 (extracelular con lagunas)
bvtv_lac   = phi
bvtv_ultra = 1.0 - bvtv_lac

comm = mesh.comm
rank = comm.rank

# Etapa 4 (ultra): extrafibrilar + fibrilla
# bvtv_fib  = nu_o + nu_m
# bvtv_ef   = nu_m + nu_w
# Etapa 5 (extracelular con lagunas)
# bvtv_lac   = array_phi
# bvtv_ultra = 1.0 - bvtv_lac


# Número de celdas
n_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
n_cells_local  = mesh.topology.index_map(mesh.topology.dim).size_local
n_vertices_global = mesh.topology.index_map(0).size_global
n_vertices_local  = mesh.topology.index_map(0).size_local

P_cyl_col = np.array([
    [0.068, -0.030, 0.4125, 0, 0, 0],
    [-0.030, 0.068, 0.4125, 0, 0, 0],
    [0.4125, 0.4125, 0.068, 0, 0, 0],
    [0, 0, 0, 0.4125, 0, 0],
    [0, 0, 0, 0, 0.4125, 0],
    [0, 0, 0, 0, 0, 0.048]
])


# Crear arrays (uno por tipo de valor)
bvtv_im = 0.43
bvtv_wetcol = 0.3

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim, 0)
connectivity = mesh.topology.connectivity(tdim, 0)

n_cells = connectivity.num_nodes
bvtv_HA_array = np.zeros((n_cells, 6, 6))

bvtv_im_array = np.zeros((n_cells, 6, 6))
bvtv_wetcol_array = np.zeros((n_cells, 6, 6))
bvtv_HA_array = np.zeros((n_cells, 6, 6))
bvtv_ic_array = np.zeros((n_cells,6,6))

bvtv_fib_array = np.zeros((n_cells,6,6))
bvtv_ef_array = np.zeros((n_cells,6,6))

bvtv_lac_array = np.zeros((n_cells,6,6))
bvtv_phi_array =np.zeros((n_cells,6,6)) 

def get_bvtv_wetcol_per_cell(bvtv_wetcol):
    arr = np.asarray(bvtv_wetcol)
    # caso escalar
    if arr.ndim == 0 or arr.size == 1:
        return np.full(n_cells_local, float(arr), dtype=np.float64)
    # caso ya por celda (global)
    if arr.size == n_cells_global:
        # extraer la porción local (suponiendo particionamiento contiguo)
        counts = comm.allgather(n_cells_local)
        start_cell = sum(counts[:comm.rank])
        stop_cell = start_cell + n_cells_local
        return arr[start_cell:stop_cell].astype(np.float64)
    # caso por nodo (global)
    if arr.size == n_vertices_global:
        # convertimos nodo->celda por promedio
        if not mesh.topology.has_connectivity(mesh.topology.dim, 0):
            mesh.topology.create_connectivity(mesh.topology.dim, 0)
        conn = mesh.topology.connectivity(mesh.topology.dim, 0)
        verts_per_cell = conn.index_map_bs
        cell_vertices = conn.array.reshape((-1, verts_per_cell))  # shape (n_cells_global, verts_per_cell)
        # calcular por celda, luego extraer porción local
        per_cell_global = np.mean(arr[cell_vertices], axis=1)  # len = n_cells_global
        counts = comm.allgather(n_cells_local)
        start_cell = sum(counts[:comm.rank])
        stop_cell = start_cell + n_cells_local
        return per_cell_global[start_cell:stop_cell].astype(np.float64)
    # caso arr es local ya (n_cells_local)
    if arr.size == n_cells_local:
        return arr.astype(np.float64)
    # caso inesperado -> fallback: truncar/pad
    if rank == 0:
        print(f"[WARN] bvtv_wetcol tamaño inesperado ({arr.size}). Se aplicará truncado/padding a n_cells_local={n_cells_local}.")
    out = np.zeros(n_cells_local, dtype=np.float64)
    length = min(arr.size, n_cells_local)
    out[:length] = arr.ravel()[:length]
    return out

# ---------- Obtener f_col por celda (local) ----------
f_col_cell = get_bvtv_wetcol_per_cell(bvtv_wetcol)  # array len = n_cells_local
if rank == 0:
    print(f"[INFO] bvtv_wetcol per-cell: local len={f_col_cell.size}, global cells={n_cells_global}")


# ====================================================
# Aca se capturan los valores de las propiedades
#=====================================================
for i, tag in enumerate(ct.values):
    # Acá se cambió el valor del agua intermolecular 
    # es la matriz de datos de la CT
    # Siempre escalar, constante para todas las celdas im=0.43  
    # bvtv_im_array[i, :, :] = np.eye(6) * bvtv_im 
    # Escalar o vector orgánico=0.31
    bvtv_wetcol_array[i, :, :] = np.eye(6) * bvtv_wetcol
    # Vector variable

for i in range(n_cells):
    # Obtener los nodos de la celda i
    # Promediar el valor de los nodos
    node_ids = connectivity.links(i)
    val_HA = np.mean(bvtv_HA_in[node_ids])
    val_ic = np.mean(bvtv_ic[node_ids])
 
    
    # Asignar el valor a la matriz 6x6
    bvtv_HA_array[i, :, :] = np.eye(6) * val_HA
    bvtv_ic_array[i,:,:]=np.eye(6)* val_ic
    bvtv_fib_array[i,:,:] = val_HA + bvtv_wetcol*np.eye(6)
    bvtv_ef_array[i,:,:] = val_HA + val_ic*np.eye(6)
    bvtv_phi_array[i,:,:] = phi*np.eye(6) 
    
bvtv_total = np.stack([bvtv_im_array, bvtv_wetcol_array, bvtv_HA_array, bvtv_ic_array, bvtv_ef_array, bvtv_fib_array, bvtv_phi_array], axis=0)




# Agua (bulk)
GPa_to_Nmm2=1
TOL = 1e-9
EPS = 1e-6
# Tamaños de malla locales (owned)
C_dim = 36
n_owned = mesh.topology.index_map(mesh.topology.dim).size_local*C_dim
n_ok    = n_owned  # o menor si estás particionando el trabajo por bloques

def symm6(C):
    return 0.5*(C + C.T)

def safe_inv(A, rcond=1e-12):
    """Inversión estable para matrices 6x6: añade tol en la diagonal si es necesario."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape != (6,6):
        raise ValueError(f"safe_inv espera una matriz 6x6, recibió shape={A.shape}")
    # pequeña regularización
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.inv(A + rcond * np.eye(6))


def J_tensor():
    """Proyector esférico (Voigt 6x6)"""
    J = np.zeros((6,6))
    J[0:3,0:3] = 1/3
    return J

def K_tensor():
    I = np.eye(6)
    return I - J_tensor()

J = J_tensor()
K = K_tensor()
I6 = np.eye(6)



# ==== Isotropizar (k,mu) y construir C isotrópico ====
def C_isotropic(k, mu):
    """C = 3k J + 2mu K"""
    return 3.0*k*J + 2.0*mu*K

# ==== Voigt/Reuss/Hill genéricos ====
def KV_from_C(C):
    return (C[0,0]+C[1,1]+C[2,2] + 2*(C[0,1]+C[0,2]+C[1,2])) / 9.0

def GV_from_C(C):
    return (C[0,0]+C[1,1]+C[2,2] - (C[0,1]+C[0,2]+C[1,2]) + 3*(C[3,3]+C[4,4]+C[5,5])) / 15.0

def KR_from_S(S):
    return 1.0 / (S[0,0]+S[1,1]+S[2,2] + 2*(S[0,1]+S[0,2]+S[1,2]))

def GR_from_S(S):
    return 15.0 / (4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[0,2]+S[1,2]) + 3*(S[3,3]+S[4,4]+S[5,5]))

def EH_nuH_from_C(C):
    S = safe_inv(C)
    KV = KV_from_C(C); GV = GV_from_C(C)
    KR = KR_from_S(S); GR = GR_from_S(S)
    KH = 0.5*(KV+KR); GH = 0.5*(GV+GR)
    E  = (9*KH*GH)/(3*KH+GH)
    nu = (3*KH-2*GH)/(2*(3*KH+GH))
    return KH, GH, E, nu

# ===== Eshelby/Hill para inclusiones esféricas en matriz isotrópica =====
def hill_alpha_beta(km, mum):
    """Coeficientes del tensor de Hill en 6x6 (esfera en matriz isotrópica)"""
    alpha = (3*km) / (3*km + 4*mum)
    beta  = (6*(km + 2*mum)) / (5*(3*km + 4*mum))
    return alpha, beta

def P_sph_influence(C0_voigh):
    """
    P para inclusión esférica: P = S_m @ Cm^{-1}, con S_m = alpha J + beta K,
    usando (km, mum) de la matriz isotropizada por Hill.
    """
    km, mum, _, _ = EH_nuH_from_C(C0_voigh)  # usamos KH, GH como isotropización
    alpha, beta = hill_alpha_beta(km, mum)
    S_m = alpha*J + beta*K
    return S_m @ safe_inv(C0_voigh)

# ===== Mori–Tanaka genérico (una fase inclusa en matriz) ===============
# -- Código revisado a 6_10_25
#  COrreccion de MT_one shot
# Correción de nombres de variables y de operaciones C_i, C_m, P_cyl_col
# ======================================================================
# Recoger la cantidad de agua para la etapa 1------------------------
# Convierto de valores por nodo a valores por celda para calcular MT_one_shot
# ========================================================================


def MT_one_shot(Cmat, fim, P_cyl_col, C_H2O):
    """
    Mori-Tanaka (una familia de inclusiones):
    C_eff = ( (1-fi) Cm + fi Ci @ A_i ) @ inv( (1-fi) I + fi A_i ),
    A_i = [ I + P @ (Ci - Cm)]^{-1}
    """
    A_i = safe_inv(I6 + P_cyl_col @ (C_H2O - Cmat))
    num = (1.0 - fim) * Cmat + fim * (Cmat @ A_i)
    den = safe_inv((1.0 - fim) * I6 + fim * A_i)
    return num @ den


# Hidroxiapatita (isótropa)
k_HA  = 82.6 * GPa_to_Nmm2
mu_HA = 44.9 * GPa_to_Nmm2
C_HA  = C_isotropic(k_HA, mu_HA)

def _reg_inv(A, reg=1e-12):
    # inversa regularizada (evita singularidad)
    return np.linalg.inv(A + reg*np.eye(A.shape[0]))


I6 = np.eye(6)
Jv = np.zeros((6,6)); Jv[:3,:3] = 1.0/3.0   # J en Voigt
Kv = I6 - Jv


# Matrices de influencia tabuladas para inclusiones cilíndricas (adimensionales)
# (Usa tus valores de referencia que ya manejas)

# -----------------------------------------------------------------
# --- MATRICES CLAVE (Voigt 6x6) --- C_MT_wetcol_06_11_25
#  Esta matriz es el punto de partida con el que termina la etapa 1
# Esta matriz es producto de la corrección del valor. Presenta una desviacíon
# en la simetría del cálculo - pero lo voy a dejar  y evalua resultados.
# La matriz deberia ser simétrica !!!!!!
# -----------------------------------------------------------------
# C_wetcol = np.array([
#  [6.317, 3.503, 4.289, 0.,    0.,    0.   ],
#  [3.503, 6.317, 4.289, 0.,    0.,    0.   ],
#  [4.227, 4.227, 9.017, 0.,    0.,    0.   ],
#  [0.,    0.,    0.,    1.325, 0.,    0.   ],
#  [0.,    0.,    0.,    0.,    1.325, 0.   ],
#  [0.,    0.,    0.,    0.,    0.,    1.407]
# ], dtype=float)



# def MT_two_families(C0, Ci_list, P_list, f_list, max_iter=80, tol=1e-8, mix=0.6):
#     # MT iterativo con dos familias (generalizable)
#     I6 = np.eye(6)
#     Ceff = C0.copy()
#     for _ in range(max_iter):
#         A_list = []
#         for Ci, P in zip(Ci_list, P_list):
#             Pm = P(Ceff) if callable(P) else P
#             A_list.append(np.linalg.inv(I6 + Pm @ (Ci - Ceff)))
#         Num = sum(f * (Ci @ Ai) for f, Ci, Ai in zip(f_list, Ci_list, A_list))
#         Den = sum(f * Ai       for f, Ai       in zip(f_list, A_list))
#         Cnew = Num @ np.linalg.inv(Den)
#         Cnext = mix*Cnew + (1.0 - mix)*Ceff
#         rel = np.linalg.norm(Cnext - Ceff)/max(np.linalg.norm(Ceff),1e-16)
#         Ceff = 0.5*(Cnext + Cnext.T)
#         if rel < tol:
#             break
#     return Ceff

# Función para calcular D_2
def calculate_D2(c_1111, c_1122):
    return c_1111 - c_1122

# Funciones para calcular las componentes de P_cyl
def P_cyl_1111(c_1111, c_1122):
    D_2 = calculate_D2(c_1111, c_1122)
    return (1/8) * (5 * c_1111 - 3 * c_1122) / (c_1111 * D_2)

def P_cyl_1122(c_1111, c_1122):
    D_2 = calculate_D2(c_1111, c_1122)
    return -(1/8) * (c_1111 + c_1122) / (c_1111 * D_2)

def P_cyl_2323(c_2323):
    return (1/8) * c_2323

def P_cyl_1212(c_1111, c_1122):
    D_2 = calculate_D2(c_1111, c_1122)
    return (1/8) * (3 * c_1111 - c_1122) / (c_1111 * D_2)


# ----------------- Evaluador de P_sph desde C0 (Voigt 6x6) -----------------
#def compute_P_sph_from_C0(C0_voigt, ngq=120, eps_guard=1e-14):
    

 ## ------------------------------------------------------------------------------
 ## ------------------------------------------------------------------------------
 ## Cálculo de la componente esférica de 
 ## ------------------------------------------------------------------------------
 ## ------------------------------------------------------------------------------
 
def compute_P_sph_from_C0_adaptive_quad(C0_voigt, epsabs=1e-8, epsrel=1e-6, eps_guard=1e-16, ngq_fallback=160):
    """
    Calcula P1122, P1133, P2323, P3333 usando integración adaptativa (scipy.integrate.quad)
    sobre x in [-1,1]. Si quad falla para una componente, usa cuadratura Gauss-Legendre fallback.
    """

    C0 = np.asarray(C0_voigt, dtype=np.float64)
    if C0.shape != (6,6):
        raise ValueError("C0_voigt debe ser 6x6 (Voigt).")

    # extraer constantes
    C1111 = float(C0[0,0]); C1122 = float(C0[0,1]); C1133 = float(C0[0,2])
    C2323 = float(C0[3,3]); C3333 = float(C0[2,2])
    C1133_sq = C1133*C1133; C1111_sq = C1111*C1111; C2323_sq = C2323*C2323

    # denominadores (funciones de x)
    def D1(x):
        x2 = x*x; x4 = x2*x2; x6 = x4*x2
        return (
            -2.0 * C1111_sq * x4 * C3333 + 2.0 * C2323_sq * x6 * C3333
            - 4.0 * C1111 * C2323_sq * x4 - 3.0 * C1111_sq * C2323 * x2
            + C1111_sq * x2 * C3333 + 2.0 * C1111 * C1133_sq * x2
            - 2.0 * C2323_sq * x4 * C1133 + 2.0 * C1111_sq * x6 * C1133
            + 4.0 * C2323 * C1133 * x2 - 2.0 * C1111 * C1133_sq * x4
            + 2.0 * C1111_sq * x6 * C1133 + C1111_sq * C2323
            - C1111_sq * C1133 * x2 + 3.0 * C1111 * C2323 * C3333
            + C1111 * C2323_sq * x2 - 3.0 * C1111 * C2323 * C3333 * x4
            + 3.0 * C1111 * C1133 * C3333 + C1111 * x6 * C2323 * C3333
        )

    def D2(x):
        x2 = x*x; x4 = x2*x2
        return (
            2.0 * C2323 * x4 * C1133 + C2323 * x4 * C3333 + C1111 * x4 * C2323
            - 2.0 * C2323 * x2 * C1133 - 2.0 * C1111 * C2323 * x2 + C1111 * C2323
            + x4 * C1133_sq - C1111 * x4 * C3333 - x2 * C1133_sq + C1111 * x2 * C3333
        )

    # numeradores (funciones de x) según tus fórmulas
    def num_1122(x):
        x2 = x*x; x4 = x2*x2
        return (
            C1111 * C2323 - 2.0*C1111*C2323*x2 + C1111*x2*C3333 + C1122*C2323
            - 2.0*C2323*x2*C1133 + 4.0*C2323*x4*C1133 - 2.0*x2*C1133_sq + 2.0*x4*C1133_sq
            - 4.0*C2323*x2*C1133 + C1122*x2*C3333 - C1111*x2*C3333
            + C1111*x4*C2323 - C1111*x4*C3333 + C1122*x4*C2323 - C1122*x4*C3333
            - 2.0*C2323_sq*x2
        )

    def num_1133(x):
        x2 = x*x
        return (-1.0 + x2) * x2 * (C2323 + C1133)

    def num_2323(x):
        x2 = x*x; x4 = x2*x2; x6 = x4*x2
        return (
            4.0*C1111*C2323*x2 - 8.0*C2323*x4*C1133 - 2.0*x4*C1133_sq - C1122*x4*C3333
            - 8.0*C1111*x4*C2323 + 3.0*C1111*x4*C3333 + 4.0*C1111*x4*C1133
            - 4.0*C1122*x4*C1133 + 2.0*C1122*x6*C1133 - 2.0*C1111*x6*C1133
            + C1122*x6*C1111 - 3.0*C1122*x4*C1111 + 3.0*C1122*C1111*x2
            - 2.0*C1111*x2*C1122 + C1122*x2*C1133 + 8.0*x6*C2323*C1133
            - 3.0*x6*C1111*C3333 + 4.0*x6*C2323*C3333 + 4.0*C1111*x6*C2323
            + C1122*x6*C3333 + 3.0*C1111*x4*C1133_sq - C1111_sq*x6
            + 2.0*C1133*x6 - 3.0*C1111_sq*x2 + C1111_sq - C1122*C1111
        )

    def num_3333(x):
        x2 = x*x
        return x2 * (x2 * C2323 - C1111 * x2 + C1111)

    # integrandos finales (protegemos denominadores con eps_guard)
    def integrand_1122(x):
        d = D1(x)
        if abs(d) < eps_guard:
            d = eps_guard if d >= 0 else -eps_guard
        return (1.0/16.0) * num_1122(x) * (1.0 - x*x) / d

    def integrand_1133(x):
        d = D2(x)
        if abs(d) < eps_guard:
            d = eps_guard if d >= 0 else -eps_guard
        return 0.25 * num_1133(x) / d

    def integrand_2323(x):
        d = D1(x)
        if abs(d) < eps_guard:
            d = eps_guard if d >= 0 else -eps_guard
        return (1.0/16.0) * num_2323(x) / d

    def integrand_3333(x):
        d = D2(x)
        if abs(d) < eps_guard:
            d = eps_guard if d >= 0 else -eps_guard
        return 0.5 * num_3333(x) / d

    # helper que usa quad y hace fallback a Gauss si algo falla
    def integrate_with_quad(fun, a=-1.0, b=1.0):
        try:
            val, err = quad(fun, a, b, epsabs=epsabs, epsrel=epsrel, limit=200)
            if not np.isfinite(val):
                raise RuntimeError("quad returned non-finite")
            return float(val), float(err)
        except Exception as e:
            # fallback: gauss with ngq_fallback points
            from numpy.polynomial.legendre import leggauss
            xg, wg = leggauss(ngq_fallback)
            # map xg from [-1,1] already
            vals = fun(xg) if hasattr(fun, "__call__") and np.ndim(xg)>0 else np.array([fun(xx) for xx in xg])
            # if fun can accept vectorized xg, use it; else fallback loop:
            if np.ndim(vals) == 0:
                vals = np.array([fun(xx) for xx in xg])
            return float(np.sum(wg * vals)), float(np.nan)

    # finalmente integrar cada componente
    P1122_val, P1122_err = integrate_with_quad(integrand_1122)
    P1133_val, P1133_err = integrate_with_quad(integrand_1133)
    P2323_val, P2323_err = integrate_with_quad(integrand_2323)
    P3333_val, P3333_err = integrate_with_quad(integrand_3333)

    P_dict = {
        "P1122": P1122_val,
        "P1133": P1133_val,
        "P2323": P2323_val,
        "P3333": P3333_val,
        "err": {"P1122": P1122_err, "P1133": P1133_err, "P2323": P2323_err, "P3333": P3333_err}
    }
    return P_dict
    
    
  

# ---------------------------- Helpers: ensamblador P (opción A) ----------------------------
#import numpy as np

def assemble_P_voigt_from_P_dict_full(Pd):
    """
    Ensambla un tensor P (Voigt 6x6) a partir de las componentes devueltas por
    compute_P_sph_from_C0 (P1122, P1133, P2323, P3333).
    La asamblea respeta simetrías transversely-isotropic (1<->2) y asigna valores
    razonables a las entradas diagonales y de corte.
    Retorna P (6x6 numpy array).
    """
    # Extraer componentes esperadas (si faltan, usar 0.0)
    P1122 = float(Pd.get("P1122", 0.0))
    P1133 = float(Pd.get("P1133", 0.0))
    P2323 = float(Pd.get("P2323", 0.0))
    P3333 = float(Pd.get("P3333", 0.0))

    P = np.zeros((6,6), dtype=np.float64)

    # Simetrías básicas:
    # normal-normal couplings:
    P[0,1] = P[1,0] = P1122
    P[0,2] = P[2,0] = P1133
    P[1,2] = P[2,1] = P1133

    # Diagonales normales: asignación razonable a partir de acoplamientos
    # (si luego quieres fórmulas exactas sustituimos aquí)
    # idea: P11,P22 promedio de sus acoplamientos con otras normales
    P11 = (P1122 + P1133) / 2.0
    P22 = P11
    P33 = P3333
    P[0,0] = P11
    P[1,1] = P22
    P[2,2] = P33

    # Cortantes: usar P2323 en 44,55,66 (Voigt convention)
    P[3,3] = P2323
    P[4,4] = P2323
    P[5,5] = P2323

    # Otras entradas cruzadas que suelen ser pequeñas en el caso esférico y TI
    # las dejamos en 0 para no introducir asimetrías no justificadas.
    # Asegurar simetría numérica
    P = 0.5 * (P + P.T)

    return P

def C_iso(K, G):
    return np.array([
        [K + 4.0/3.0*G, K - 2.0/3.0*G, K - 2.0/3.0*G, 0., 0., 0.],
        [K - 2.0/3.0*G, K + 4.0/3.0*G, K - 2.0/3.0*G, 0., 0., 0.],
        [K - 2.0/3.0*G, K - 2.0/3.0*G, K + 4.0/3.0*G, 0., 0., 0.],
        [0., 0., 0., G, 0., 0.],
        [0., 0., 0., 0., G, 0.],
        [0., 0., 0., 0., 0., G]
    ], dtype=np.float64)

# ---------------------------- Caching util ----------------------------
from collections import OrderedDict
class LRUCache:
    def __init__(self, maxsize=1024):
        self.maxsize = maxsize
        self._od = OrderedDict()
    def get(self, k):
        v = self._od.get(k, None)
        if v is not None:
            self._od.move_to_end(k)
        return v
    def set(self, k, v):
        self._od[k] = v
        self._od.move_to_end(k)
        if len(self._od) > self.maxsize:
            self._od.popitem(last=False)

# cache global para P por C_eff "prácticamente iguales"
P_cache = LRUCache(maxsize=2048)

def _C_key(C, tol=1e-8, ndigits=8):
    """Crea una clave hashable para una matriz 6x6 C; redondea a ndigits para agrupar similares."""
    flat = np.asarray(C, dtype=np.float64).ravel()
    # redondear para estabilidad y convertir a tuple
    rounded = tuple(np.round(flat, ndigits).tolist())
    return rounded

# ---------------------------- build_P_list (opción C integrada) ----------------------------


# ---------------------------- Ejemplo de integración en bucle por celdas (MPI-friendly) ----------------------------
# Supone que tienes C_eff_local (n_cells_local,6,6) con la rigidez efectiva calculada por celda.
# Llamará a build_P_list_from_Ceff por celda y devolverá P_list_local: lista de parejas por celda.


# ---------------------------- Uso típico (etapa C: integración en MT loop) ----------------------------
# Ejemplo de bucle principal que obtiene P_list y llama MT_two_families por celda:
#
# C_eff_local : (n_cells_local,6,6)  <-- obtenido previamente (resultado de alguna etapa)
# Ci_list_local per cell (por ejemplo [C_fibrilla_local[ic], C_foam_local[ic]])
# f_list_local per cell  (por ejemplo [f_fibril_cell[ic], f_extraf_cell[ic]])
#
# P_pairs_local = compute_P_list_for_cells(C_eff_local, ngq=160)
# for ic in range(n_cells_local):
#     P_cyl_mandel, P_sph_mandel = P_pairs_local[ic]
#     # construir un P_list_builder local sencillo que devuelve estos dos tensores cuando se llama
#     def P_list_builder_local(Ceff_unused=None, P_cyl=P_cyl_mandel, P_sph=P_sph_mandel):
#         # Ceff_unused ignorado: esta versión usa los P precomputados (const por celda)
#         return [P_cyl, P_sph]
#     # ahora llamas a MT_two_families con Ci_list_local, f_list_local y P_list_builder_local
#     Ceff_result = MT_two_families(C0_local, Ci_list_local, P_list_builder_local, f_list_local, ...)
#     ...

# ----------------- Ejemplo de uso -----------------
# si tienes C0_voigt (6x6):
# Pvals = compute_P_sph_from_C0(C0_voigt, ngq=160)
# print(Pvals)



# -----------------------------------------
# 2) Buffers por etapa (C 6x6 aplanado)
# -----------------------------------------
Cwet_local_flat  = np.zeros(n_owned*C_dim)  # Etapa 1
Cfib_local_flat  = np.zeros(n_owned*C_dim)  # Etapa 2
Cef_local_flat   = np.zeros(n_owned*C_dim)  # Etapa 3
Cultra_local_flat= np.zeros(n_owned*C_dim)  # Etapa 4
Cvasc_local_flat = np.zeros(n_owned*C_dim)  # Etapa 5


# =========================
# Etapa 1 — C_MT_wetcol por celda + salida XDMF
# =========================
# --- datos etapa 1 ---
###################---------------------------------- VIP--------------------------

V = fem.functionspace(mesh, ("DG", 0))
f_bvtv_HA = fem.Function(V,name="bvtv_HA_in")
f_bvtv_wetcol  = fem.Function(V,name="bvtv_wetcol")
f_bvtv_im = fem.Function(V,name="bvtv_im")
f_bvtv_ic = fem.Function(V,name="bvtv_ic")
f_bvtv_fib = fem.Function(V,name="bvtv_fib")
f_bvtv_ef = fem.Function(V,name="bvtv_ef")



# Promedio de cada matriz (valor escalar por celda)
f_bvtv_HA.x.array[:] = np.mean(bvtv_HA_array[:, range(6), range(6)], axis=1)
f_bvtv_wetcol.x.array[:]= np.mean(bvtv_wetcol_array[:,range(6), range(6)], axis=1)
f_bvtv_im.x.array[:]= np.mean(bvtv_im_array[:,range(6), range(6)], axis=1)
f_bvtv_ic.x.array[:]= np.mean(bvtv_ic_array[:,range(6), range(6)], axis=1)
f_bvtv_fib.x.array[:]= np.mean(bvtv_fib_array[:,range(6), range(6)], axis=1)
f_bvtv_ef.x.array[:]= np.mean(bvtv_ef_array[:,range(6), range(6)], axis=1)

# if MPI.COMM_WORLD.rank == 0:
#     with XDMFFile(MPI.COMM_WORLD, "./17_11_25/bvtv_HA_in.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         xdmf.write_function(f_bvtv_HA)
    
#     with XDMFFile(MPI.COMM_WORLD, "./17_11_25/bvtv_wetcol.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         xdmf.write_function(f_bvtv_wetcol)
        
#     with XDMFFile(MPI.COMM_WORLD, "./17_11_25/bvtv_im.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         xdmf.write_function(f_bvtv_im)



# φ_im por celda: constante 0.43
#phi_im_cell = fem.Function(V0, name="phi_im_cell")
# Valor de la porosidad en el colágeno  .... A valor constante _0.43 con este valor 
# inicia el modelo.
###################---------------------------------- VIP--------------------------
# Colágeno seco (GPa)
C_col_dry = np.array([
    [11.7,  5.1,  7.1,  0.0,  0.0,  0.0],
    [ 5.1, 11.7,  7.1,  0.0,  0.0,  0.0],
    [ 7.1,  7.1, 17.9,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  3.3,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  3.3,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  3.3]
], dtype=float)

# Agua (isotrópica) solo en volumétrico: C = 3 K J
K_H2O = 2.3  # GPa
J = np.zeros((6,6)); J[:3,:3] = 1/3
C_im = 3.0*K_H2O*J

# Tensor de influencia (cilíndrico “simple”, estable)
P_cyl_col = np.array([
    [ 0.068, -0.030,  0.4125, 0, 0, 0],
    [-0.030,  0.068,  0.4125, 0, 0, 0],
    [ 0.4125, 0.4125, 0.068,  0, 0, 0],
    [ 0,      0,      0,      0.4125, 0, 0],
    [ 0,      0,      0,      0,      0.4125, 0],
    [ 0,      0,      0,      0,      0,      0.048]
], dtype=float)

I6 = np.eye(6)
A_im = _reg_inv(I6 + P_cyl_col @ (C_im - C_col_dry))  # fijo (host=col_dry)

# def C_MT_wetcol_cell(phi_im: float) -> np.ndarray:
#     f_c = 1.0 - bvtv_im_array
#     f_w = bvtv_im_array
#     Num = f_w*(C_im @ A_im) + f_c*(C_col_dry @ I6)
#     Den = f_w*A_im + f_c*I6
#     return symm6(Num @ _reg_inv(Den))

# --- buffers y loop ---


# ---------- Propiedades (unidades N/mm^2 = MPa) ----------
K_H2O = 2.3     # agua (matriz)
G_H2O = 0.0

K_col = 42.64   # colágeno seco (inclusión)
G_col = 2.17

# ---------- util: C_iso (Voigt 6x6) ----------

C_H2O = C_iso(K_H2O, G_H2O)
C_col  = C_iso(K_col, G_col)
I6 = np.eye(6, dtype=np.float64)

# ---------- Eshelby-like / P_sph (simplificada para inclusiones esféricas) ----------
# versión simple: S with 1/3 on normal-normal part
P_sph = np.zeros((6,6), dtype=np.float64)
#P_sph[:3, :3] = np.eye(3) * (1.0/3.0)
# (si tienes P_sph más preciso, sustituye esta matriz)

# ---------- Helper: obtener scalar por celda desde bvtv_wetcol (soporta escalar, por-celda, por-nodo) ----------


# ---------- Calcular A_col (corrección solicitada) y C_MT por celda ----------
# Usamos la versión: A_col = inv(I6 + P_sph @ (C_col - C_H2O))
# luego C_MT = C_H2O + f_col * (C_col - C_H2O) @ A_col

# Pre-calcular matriz fija A_col_base (si C_col,C_H2O fijos)
# Intentamos evitar invertir por cada celda si P_sph y Ci-Cm constantes.
M_base = I6 + P_cyl_col @ (C_col - C_H2O)
# chequeo invertibilidad
try:
    A_col_base = np.linalg.inv(M_base)
except np.linalg.LinAlgError:
    # regularizar con pequeña diagonal eps
    
    eps = 1e-8
    if rank == 0:
        print("[WARN] M_base singular; aplicando regularización eps on diagonal.")
    A_col_base = np.linalg.inv(M_base + eps * np.eye(6))
##### =================================================================================
##### =================================================================================
##  --------------  Etapa 1. ---------------------------------------------------------
##### =================================================================================
##### =================================================================================

# crear array local para C_MT
C_MT_wetcol_cell = np.zeros((n_cells_local, 6, 6), dtype=np.float64)

for ic in range(n_cells_local):
    fbvtv_ic = bvtv_ic_array[ic, :, :]  # matriz 6x6, no float
    fim = fbvtv_ic.copy()
    Cmat = C_col.copy()

    C_MT_wetcol_cell[ic, :, :]= MT_one_shot(Cmat, fim, P_cyl_col, C_H2O)


# ---------- Informes y chequeos ----------
local_mins = np.min(C_MT_wetcol_cell.reshape(n_cells_local, -1), axis=1)
local_maxs = np.max(C_MT_wetcol_cell.reshape(n_cells_local, -1), axis=1)
glob_min = comm.allreduce(np.min(local_mins), op=MPI.MIN)
glob_max = comm.allreduce(np.max(local_maxs), op=MPI.MAX)


# ========= Crear un espacio escalar por celda =========
#V = fem.functionspace(mesh, ("DG", 0))

# Escogemos una métrica escalar para visualizar (ejemplo: traza o C_11)
# 1️⃣ traza: promedio de C_xx, C_yy, C_zz
#trace_C = np.mean(C_MT_wetcol_cell[:, [0,1,2], [0,1,2]], axis=1)

# 2️⃣ o si prefieres C_11 (componente axial)
trace_C = C_MT_wetcol_cell[:, 0, 0]



# 1️⃣ Calcular baricentros de celdas
mesh_coords = mesh.geometry.x
connectivity = mesh.topology.connectivity(mesh.topology.dim, 0)
cell_bary = np.zeros((n_cells, 3))

for ic in range(n_cells):
    node_ids = connectivity.links(ic)
    cell_coords = mesh_coords[node_ids]
    cell_bary[ic] = np.mean(cell_coords, axis=0)

# 2️⃣ Tomar el valor escalar/tensorial de cada celda
values = C_MT_wetcol_cell[:, 0, 0]  # ejemplo: componente (0,0)

# 3️⃣ Interpolar a nodos
interp = LinearNDInterpolator(cell_bary, values)
node_values = interp(mesh_coords)
node_values = np.nan_to_num(node_values, nan=np.nanmean(node_values))

# Crear función escalar
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
C11 = dolfinx.fem.Function(V, name="C_Mt_Wetcol")

# Asignar valores a los nodos
C11.x.array[:] = node_values.flatten()

# Guardar para Paraview *** Aca estamos imprimendo nodos.
if rank == 0:
    with XDMFFile(MPI.COMM_WORLD, "./17_11_25/C_MT_wetcol.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(C11)
if rank == 0:
    print(f"[OK] C_MT_wetcol_cell construido. Componente rango global: min={glob_min:.6g}, max={glob_max:.6g}")


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# # ========= Crear función fenicsx y asignar para celdas =========
# f_Cwetcol = fem.Function(V,name="C_wetcol")
# # Asumimos que n_cells_local == len(trace_C)
# f_Cwetcol.x.array[:] = trace_C.astype(np.float64)

# # ========= Guardar a archivo XDMF =========
# output_file = "./17_11_25/C_MT_wetcol_trace.xdmf"
# # Guardar campo solo en el proceso principal
# if MPI.COMM_WORLD.rank == 0:
#     with XDMFFile(mesh.comm, output_file, "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         xdmf.write_function(f_Cwetcol)
#     print(f"[OK] Exportado C_MT_wetcol_cell → {output_file} (ParaView listo).")

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------



# ejemplo: imprime stats básicos de la etapa 1
# if MPI.COMM_WORLD.rank == 0:
#     # toma unas cuantas celdas
#     sample = min(10, n_ok_e1)
#     Ks, Gs, Es, nus = [], [], [], []
#     for c in range(sample):
#         Cw = Cwet_local_flat[c*C_dim:(c+1)*C_dim].reshape(6,6)
#         KH, GH, Eiso, nuiso = EH_nuH_from_C(Cw)
#         Ks.append(KH); Gs.append(GH); Es.append(Eiso); nus.append(nuiso)
#     print(f"[Etapa1] Ejemplo primeras {sample} celdas — K~{np.mean(Ks):.3g} GPa, G~{np.mean(Gs):.3g} GPa, E~{np.mean(Es):.3g} GPa, nu~{np.mean(nus):.3f}")
    
# -------------------------------------------------------
# Etapa 2 — Fibrilla (colágeno húmedo + HA intrafibrilar)
#     Inclusiones: cilíndrica (colágeno húmedo) + esférica/cilíndrica (HA_in)
# -------------------------------------------------------
# ============================
# ETAPA 2 — Fibrilla (MT 2-familias: colágeno húmedo cilíndrico + HA esférica)
# ============================


# ---------- C_iso(K,G) (Voigt) ----------
def C_iso_from_KG(Kb, Gs):
    lam = Kb - 2.0*Gs/3.0
    C = np.zeros((6,6))
    C[0,0]=C[1,1]=C[2,2]=lam+2*Gs
    C[0,1]=C[0,2]=C[1,0]=C[1,2]=C[2,0]=C[2,1]=lam
    C[3,3]=C[4,4]=C[5,5]=Gs
    return C


def build_P_cyl_col_from_Cvoigh(C0_voigt, eps=1e-12):
    """
    Construye P_cyl^{col} (Voigt 6x6) usando las fórmulas (3)-(6) de Hellmich.
    Asegura asignación explícita de P1313 (= P2323).
    Acepta Cvoigt (6,6) o (n,6,6).
    """
    C = np.asarray(C0_voigt, dtype=float)
    single = (C.ndim == 2)
    if single:
        C = C[np.newaxis, :, :]
    n = C.shape[0]
    P_all = np.zeros((n, 6, 6), dtype=float)

    for i in range(n):
        Ci = C[i]
        # Extraer componentes relevantes (Voigt indices)
        c1111 = float(Ci[0, 0])
        c1122 = float(Ci[0, 1])
        c2323 = float(Ci[3, 3])  # c44
        # protección contra divisiones por cero
        if abs(c1111) < eps:
            c1111 = np.sign(c1111) * eps if c1111 != 0 else eps
        D2 = c1111 - c1122
        if abs(D2) < eps:
            D2 = np.sign(D2) * eps if D2 != 0 else eps
        if abs(c2323) < eps:
            c2323 = np.sign(c2323) * eps if c2323 != 0 else eps

        # Fórmulas (3)-(6)
        P1111 = (1.0 / 8.0) * (5.0 * c1111 - 3.0 * c1122) / (c1111 * D2)
        P1122 = -(1.0 / 8.0) * (c1111 + c1122) / (c1111 * D2)
        P1212 = (1.0 / 8.0) * (3.0 * c1111 - c1122) / (c1111 * D2)
        P2323 = 1.0 / (8.0 * c2323)

        P = np.zeros((6, 6), dtype=float)

        # normales / acoples (transversely isotropic)
        P[0, 0] = P[1, 1] = P1111
        P[0, 1] = P[1, 0] = P1122

        # acoplamientos radial-axial: usamos P1122 (conservador)
        P[0, 2] = P[1, 2] = P[2, 0] = P[2, 1] = P1122

        # diagonal axial provisional (puede ajustarse por anisotropía)
        P[2, 2] = P1111

        # ------- componentes de corte (Voigt) -------
        # P[3,3] -> componente 23-23 (P2323)
        # P[4,4] -> componente 13-13 (P1313) <- ¡debe ser igual a P2323 según Hellmich!
        # P[5,5] -> componente 12-12 (P1212)
        P[3, 3] = P2323   # P_{2323}
        P[4, 4] = P2323   # P_{1313} = P_{2323} según la fórmula
        P[5, 5] = P1212   # P_{1212}

        # asegurar simetría numérica
        P_all[i] = 0.5 * (P + P.T)

    return P_all[0] if single else P_all


# # prueba rápida (con valores de ejemplo)
# if __name__ == "__main__":
#     C_wetcol_example = np.array([
#         [6.317, 3.503, 4.289, 0.0,   0.0,   0.0],
#         [3.503, 6.317, 4.289, 0.0,   0.0,   0.0],
#         [4.227, 4.227, 9.017, 0.0,   0.0,   0.0],
#         [0.0,   0.0,   0.0,   1.325, 0.0,   0.0],
#         [0.0,   0.0,   0.0,   0.0,   1.325, 0.0],
#         [0.0,   0.0,   0.0,   0.0,   0.0,   1.407]
#     ], dtype=float)

#     Pcol = build_P_cyl_col_from_Cvoigt(C_wetcol_example)
#     print("P_cyl_col (Voigt 6x6):\n", np.round(Pcol, 6))
#     print("P[3,3] (2323)  =", Pcol[3,3])
#     print("P[4,4] (1313)  =", Pcol[4,4])
#     print("P[5,5] (1212)  =", Pcol[5,5])
#     print("Simétrica?:", np.allclose(Pcol, Pcol.T))

# ---------- MT para 2 familias (auto-consistente) ----------



def MT_two_families_from_fibrilla(C_MT, C0_voigt, P_cyl_voigt, C_HA, P_sph_voigt,
                            f_wetcol, f_HA, rcond=1e-12, reg_eps=1e-12):
    """
    Mori-Tanaka simplificado para dos familias (wet-collagen cilíndrico + HA esférico).
    Entradas (cada llamada trabaja con matrices 6x6 y scalars para fracciones):
      C_m            : matriz 6x6 de la matriz (wet-coll) local
      C0_voigt       : semilla Voigt 6x6 (seed)
      P_cyl    : tensor influencia cilíndrico 6x6 (Voigt)
      C_HA           : matriz 6x6 de HA
      P_sph    : tensor influencia esférico 6x6 (Voigt)
      f_wetcol       : fracción volumétrica de la inclusión/fase wetcol (scalar)
      f_HA           : fracción volumétrica de HA (scalar)
    Retorna:
      C_eff (6x6) simétrica y (con intento de) positiva definida.
    """

    # --- Validaciones / normalizaciones ---
    C_m = np.asarray(C_MT, dtype=float)
    C0 = np.asarray(C0_voigt, dtype=float)
    P_cyl = np.asarray(P_cyl_voigt, dtype=float)
    C_HA = np.asarray(C_HA, dtype=float)
    P_sph = np.asarray(P_sph_voigt, dtype=float)

    
    f_wet = np.clip(f_wetcol, 0.0, 1.0)
    f_ha  = np.clip(f_HA, 0.0, 1.0)

    # Si la suma >1, normalizamos proporcionalmente (opcional)
   
    # ---- cálculo A para cada familia (usar safe_inv que ya tienes) ----
    # A_cyl = [ I + P_cyl @ (C_i - C0) ]^{-1}  (para fibras / cilindros)
    # A_sph = [ I + P_sph @ (C_HA - C0) ]^{-1}  (esferas HA)
    A_cyl = safe_inv(I6 + P_cyl @ (C_m - C0), rcond=rcond)
    A_sph = safe_inv(I6 + P_sph @ (C_HA - C0), rcond=rcond)

    # ---- Numerador y denominador (forma Mori-Tanaka extendida, simplificada) ----
    # Aquí usamos la idea: (1 - f_tot)*C0 + f_wet * C_m @ A_cyl + f_ha * C_HA @ A_sph
    # den = (1 - f_tot)*I + f_wet * A_cyl + f_ha * A_sph
    num =  f_wet * (C_m @ A_cyl) + f_ha * (C_HA @ A_sph)
    den_mat =  f_wet * A_cyl + A_cyl + f_ha * A_sph    
    
    den_inv = safe_inv(den_mat, rcond=rcond)

    C_eff = num @ den_inv

    # Forzar simetría numérica (evita errores numéricos)
    C_eff = 0.5 * (C_eff + C_eff.T)

    # Regularización muy pequeña si aparece indefinidez numérica
    # comprobamos autovalores
    # eig = np.linalg.eigvalsh(C_eff)
    # if np.any(eig <= 0):
    #     # regularizar suavemente para forzar positivos
    #     min_eig = np.min(eig)
    #     delta = max(reg_eps, 1e-8 - min_eig)  # elevar lo suficiente
    #     C_eff += np.eye(6) * delta
    #     # asegurar simetría otra vez
    #     C_eff = 0.5 * (C_eff + C_eff.T)

    return C_eff
    

def MT_two_families_from_Ef(C_im, C_Ef, P_sph_ef, C_HA,
                            f_ic, f_HA, rcond=1e-12, reg_eps=1e-12):
    """
    Mori-Tanaka simplificado para dos familias (wet-collagen cilíndrico + HA esférico).
    Entradas (cada llamada trabaja con matrices 6x6 y scalars para fracciones):
      C_m            : matriz 6x6 de la matriz (wet-coll) local
      C0_voigt       : semilla Voigt 6x6 (seed)
      P_cyl_voigt    : tensor influencia cilíndrico 6x6 (Voigt)
      C_HA           : matriz 6x6 de HA
      P_sph_voigt    : tensor influencia esférico 6x6 (Voigt)
      f_wetcol       : fracción volumétrica de la inclusión/fase wetcol (scalar)
      f_HA           : fracción volumétrica de HA (scalar)
    Retorna:
      C_eff (6x6) simétrica y (con intento de) positiva definida.
    """

    # --- Validaciones / normalizaciones ---
    C_ic = np.asarray(C_im, dtype=float)
    C0 = np.asarray(C_Ef, dtype=float)
    P_sph_ = np.asarray(P_sph_ef, dtype=float)
    C_HA = np.asarray(C_HA, dtype=float)
    f_ic = np.clip(f_ic, 0.0, 1.0)
    f_ha  = np.clip(f_HA, 0.0, 1.0)
    f_total = f_ic + f_HA
    C_m = C0*f_total

    # Asegurar shapes
   
 


    I6 = np.eye(6)

    # ---- cálculo A para cada familia (usar safe_inv que ya tienes) ----
    # A_cyl = [ I + P_cyl @ (C_i - C0) ]^{-1}  (para fibras / cilindros)
    # A_sph = [ I + P_sph @ (C_HA - C0) ]^{-1}  (esferas HA)
    #A_cyl = safe_inv(I6 + P_cyl @ (C_m - C0), rcond=rcond)
    A_sph_HA = safe_inv(I6 + P_sph_ @ (C_HA - C_m), rcond=rcond)
    A_sph_ic = safe_inv(I6 + P_sph_ @ (C_ic - C_m), rcond=rcond)


    # ---- Numerador y denominador (forma Mori-Tanaka extendida, simplificada) ----
    # Aquí usamos la idea: (1 - f_tot)*C0 + f_wet * C_m @ A_cyl + f_ha * C_HA @ A_sph
    # den = (1 - f_tot)*I + f_wet * A_cyl + f_ha * A_sph
    num = f_ic * (C_ic @ A_sph_ic) + f_ha * (C_HA @ A_sph_HA)
    den_mat = f_ic* A_sph_ic + f_ha * A_sph_HA
    den_inv = safe_inv(den_mat, rcond=rcond)
    C_eff = num @ den_inv

    # Forzar simetría numérica (evita errores numéricos)
    C_eff = 0.5 * (C_eff + C_eff.T)

    # Regularización muy pequeña si aparece indefinidez numérica
    # comprobamos autovalores
    # eig = np.linalg.eigvalsh(C_eff)
    # if np.any(eig <= 0):
    #     # regularizar suavemente para forzar positivos
    #     min_eig = np.min(eig)
    #     delta = max(reg_eps, 1e-8 - min_eig)  # elevar lo suficiente
    #     C_eff += np.eye(6) * delta
    #     # asegurar simetría otra vez
    #     C_eff = 0.5 * (C_eff + C_eff.T)

    return C_eff



# Etapa 4 (ultra): extrafibrilar + fibrilla
# bvtv_fib  = nu_o + nu_m
# bvtv_ef   = nu_m + nu_w

# # Etapa 5 (extracelular con lagunas)
# bvtv_lac   = array_phi
# bvtv_ultra = 1.0 - bvtv_lac

   
#     # Asignar el valor a la matriz 6x6
#     bvtv_HA_array[i, :, :] = np.eye(6) * val_HA
#     bvtv_ic_array[i,:,:]=np.eye(6)* val_ic
#     bvtv_fib_array[i,:,:] = val_HA + bvtv_wetcol*np.eye(6)
#     bvtv_ef_array[i,:,:] = val_HA + val_ic*np.eye(6)
    
# bvtv_total = np.stack([bvtv_im_array, bvtv_wetcol_array, bvtv_HA_array, bvtv_ic_array, bvtv_ef_array, bvtv_fib_array], axis=0)




# Agua (bulk)

def MT_two_families_from_ExtraC(C_im, C_Ef, P_cyl_ef, C_fib,
                            f_fib, f_ef, rcond=1e-12, reg_eps=1e-12):
    """
    Mori-Tanaka simplificado para dos familias (wet-collagen cilíndrico + HA esférico).
    Entradas (cada llamada trabaja con matrices 6x6 y scalars para fracciones):
      C_m            : matriz 6x6 de la matriz (wet-coll) local
      C0_voigt       : semilla Voigt 6x6 (seed)
      P_cyl_voigt    : tensor influencia cilíndrico 6x6 (Voigt)
      C_HA           : matriz 6x6 de HA
      P_sph_voigt    : tensor influencia esférico 6x6 (Voigt)
      f_wetcol       : fracción volumétrica de la inclusión/fase wetcol (scalar)
      f_HA           : fracción volumétrica de HA (scalar)
    Retorna:
      C_eff (6x6) simétrica y (con intento de) positiva definida.
    """

    # --- Validaciones / normalizaciones ---
    C_im_ = np.asarray(C_im, dtype=float)
    P_cyl_ = np.asarray(P_cyl_ef, dtype=float)
    C_Ef_ = np.asarray(C_Ef, dtype=float)
    f_fib_ = np.clip(f_fib, 0.0, 1.0)
    f_ef_  = np.clip(f_ef, 0.0, 1.0)

    # Asegurar shapes
   


    I6 = np.eye(6)

    # ---- cálculo A para cada familia (usar safe_inv que ya tienes) ----
    # A_cyl = [ I + P_cyl @ (C_i - C0) ]^{-1}  (para fibras / cilindros)
    # A_sph = [ I + P_sph @ (C_HA - C0) ]^{-1}  (esferas HA)
    #A_cyl = safe_inv(I6 + P_cyl @ (C_m - C0), rcond=rcond)
    A_cyl_ultra = safe_inv(I6 + P_cyl_ @ (C_im_ - C_Ef_), rcond=rcond)


    # ---- Numerador y denominador (forma Mori-Tanaka extendida, simplificada) ----
    # Aquí usamos la idea: (1 - f_tot)*C0 + f_wet * C_m @ A_cyl + f_ha * C_HA @ A_sph
    # den = (1 - f_tot)*I + f_wet * A_cyl + f_ha * A_sph
    num = f_ef_ * C_Ef_ + f_fib_*(C_im_ @ A_cyl_ultra) 
    den_mat = f_ef_ * I6 +  f_fib_ * A_cyl_ultra
    den_inv = safe_inv(den_mat, rcond=rcond)
    C_eff = num @ den_inv

    # Forzar simetría numérica (evita errores numéricos)
    C_eff = 0.5 * (C_eff + C_eff.T)

    # Regularización muy pequeña si aparece indefinidez numérica
    # comprobamos autovalores
    # eig = np.linalg.eigvalsh(C_eff)
    # if np.any(eig <= 0):
    #     # regularizar suavemente para forzar positivos
    #     min_eig = np.min(eig)
    #     delta = max(reg_eps, 1e-8 - min_eig)  # elevar lo suficiente
    #     C_eff += np.eye(6) * delta
    #     # asegurar simetría otra vez
    #     C_eff = 0.5 * (C_eff + C_eff.T)

    return C_eff



def MT_two_families_from_UltraC(C_ic, C_ultra, P_sph_ultra, phi, rcond=1e-12, reg_eps=1e-12):
    """
    Mori-Tanaka simplificado para dos familias (wet-collagen cilíndrico + HA esférico).
    Entradas (cada llamada trabaja con matrices 6x6 y scalars para fracciones):
      C_ic           : matriz 6x6 de la matriz (lacunar 3K-H20-J) local
      C0_voigt       : semilla Voigt 6x6 (seed)
      P_cyl_voigt    : tensor influencia cilíndrico 6x6 (Voigt)
      C_HA           : matriz 6x6 de HA
      P_sph_voigt    : tensor influencia esférico 6x6 (Voigt)
      f_ultra          : fracción volumétrica de porosidad
    Retorna:
      C_eff (6x6) simétrica y (con intento de) positiva definida.
    """

    # --- Validaciones / normalizaciones ---
    C_lac_ = np.asarray(C_ic, dtype=float)
    P_sph_ultra_ = np.asarray(P_sph_ultra, dtype=float)
    C_ultra_ = np.asarray(C_ultra, dtype=float)
    phi_ = np.clip(phi, 0.0, 1.0)

    # Asegurar shapes
   
    I6 = np.eye(6)

    # ---- cálculo A para cada familia (usar safe_inv que ya tienes) ----
    # A_cyl = [ I + P_cyl @ (C_i - C0) ]^{-1}  (para fibras / cilindros)
    # A_sph = [ I + P_sph @ (C_HA - C0) ]^{-1}  (esferas HA)
    #A_cyl = safe_inv(I6 + P_cyl @ (C_m - C0), rcond=rcond)
    A_sph_ultra = safe_inv(I6 + P_sph_ultra_ @ (C_lac_ - C_ultra_), rcond=rcond)

    num = (1 - phi_) * C_ultra_ + phi_*(C_lac_ @ A_sph_ultra) 
    den_mat = (1 - phi_)* I6 +  phi_* A_sph_ultra
    den_inv = safe_inv(den_mat, rcond=rcond)
    C_eff = num @ den_inv

    # Forzar simetría numérica (evita errores numéricos)
    C_eff = 0.5 * (C_eff + C_eff.T)

    # Regularización muy pequeña si aparece indefinidez numérica
    # comprobamos autovalores
    # eig = np.linalg.eigvalsh(C_eff)
    # if np.any(eig <= 0):
    #     # regularizar suavemente para forzar positivos
    #     min_eig = np.min(eig)
    #     delta = max(reg_eps, 1e-8 - min_eig)  # elevar lo suficiente
    #     C_eff += np.eye(6) * delta
    #     # asegurar simetría otra vez
    #     C_eff = 0.5 * (C_eff + C_eff.T)

    return C_eff



# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------

# Etapa 2 — Espuma extrafibrilar (HA_ef + agua intercristalina, esférica)
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------

# --- información malla ---
n_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
n_cells_local  = mesh.topology.index_map(mesh.topology.dim).size_local

# --- tu array ya creado por celda (según tu snippet) ---
# bvtv_HA_array shape: (n_cells_X, 6, 6)
# puede ser global (n_cells_global,6,6) o local (n_cells_local,6,6)


# --- extraer el escalar f_HA por celda ---
# Si almacenaste np.eye(6)*val_HA, entonces trace = 6*val_HA -> val_HA = trace/6

# --- obtener f_col por celda (usa tu helper si ya la tienes; aquí lo recalculo sencillo) ---
# bvtv_wetcol puede ser escalar, global por celda o local por celda
# def get_fcol_local(bvtv_wetcol):
#     arr = np.asarray(bvtv_wetcol, dtype=np.float64).ravel()
#     if arr.size == 1:
#         return np.full(n_cells_local, float(arr), dtype=np.float64)
#     if arr.size == n_cells_global:
#         counts = comm.allgather(n_cells_local)
#         start = sum(counts[:comm.rank]); stop = start + n_cells_local
#         return arr[start:stop].astype(np.float64)
#     if arr.size == n_cells_local:
#         return arr.astype(np.float64)
#     # si viene por nodo o en otro formato, usa tu helper to_cell_array_from_input (recomendado)
#     return to_cell_array_from_input(arr)


#---------------------------------------------------------

def compute_P_cyl_voigt_from_Cvoigt(C, eps=1e-12):
    """
    C : array_like (6,6) o (n,6,6) en notación Voigt.
    Retorna P_cyl_voigt con la estructura dada por Hellmich (simétrica 6x6)
    Si C es lote, devuelve array (n,6,6).
    """
    C = np.asarray(C, dtype=float)
    single = (C.ndim == 2)
    if single:
        C = C[np.newaxis, :, :]  # shape (1,6,6)

    n = C.shape[0]
    P_all = np.zeros((n, 6, 6), dtype=float)

    for i in range(n):
        Ci = C[i]
        c11 = float(Ci[0,0])
        c12 = float(Ci[0,1])
        c44 = float(Ci[3,3])  # C_{2323} en Voigt
        D2 = c11 - c12
        # proteger divisiones
        if abs(c11) < eps or abs(D2) < eps or abs(c44) < eps:
            # si está mal condicionado, rellenar con ceros (o lanzar warning)
            # mejor devolver valores pequeños en vez de NaN
            factor_safe = max(abs(c11), eps)
            D2_safe = np.sign(D2)*max(abs(D2), eps)
            c44_safe = max(abs(c44), eps)
        else:
            factor_safe = c11
            D2_safe = D2
            c44_safe = c44

        P1111 = 0.125 * (5.0*c11 - 3.0*c12) / (factor_safe * D2_safe)
        P1122 = -0.125 * (c11 + c12) / (factor_safe * D2_safe)
        P1212 = 0.125 * (3.0*c11 - c12) / (factor_safe * D2_safe)
        P2323 = 1.0 / (8.0 * c44_safe)
        # montar matriz Voigt (simétrica)
        P = np.zeros((6,6), dtype=float)
        # normales
        P[0,0] = P1111
        P[1,1] = P1111  # P2222 = P1111
        P[0,1] = P[1,0] = P1122
        # acoplados con eje 3 (si tu formulación da otros, aquí podrías extender)
        # la imagen no da P1133 explícito; lo dejamos a 0 o lo calculas según otra relación
        # cortantes:
        P[3,3] = P2323  # 23-23
        P[4,4] = P2323  # 13-13
        P[5,5] = P1212  # 12-12

        # symmetrize por si acaso
        P = 0.5*(P + P.T)
        P_all[i] = P

# ==============================================================================
# ==============================================================================
# Etapa - 2 - fibrilla
# ==============================================================================
# ==============================================================================
# ==============================================================================

C_fibrilla_local = np.zeros((n_cells_local, 6, 6), dtype=np.float64)
Ceff_voigt = np.zeros((n_cells_local, 6, 6), dtype=np.float64)

P_sph_voigh =np.zeros((6, 6), dtype=np.float64)

HA_arr = np.asarray(bvtv_HA_array, dtype=np.float64)  # (n,6,6)
ic_arr = np.asarray(bvtv_ic_array, dtype=np.float64)  # (n,6,6)


# --- semilla global Voigt (aprox) ---
# 1) calcular P_sph_HA solo una vez (si C_HA es constante)


# --- bucle local: llamar MT_two_families por celda ---
for ic in range(n_cells_local):
   # --- extraer fracciones escalares desde la matriz 6x6 ---
    # opción A: usar trace/6 (promedio diagonal)
    fbvtv_wetcol = bvtv_wetcol_array[ic, :, :]  # matriz 6x6, no float
    fbvtv_HA = bvtv_HA_array[ic, :, :]
    f_total = fbvtv_wetcol + fbvtv_HA

    C_MT= C_MT_wetcol_cell[ic, :, :]
    C0_voigt = C_MT*f_total
    
    P_sph_CMT_dict = compute_P_sph_from_C0_adaptive_quad(C0_voigt,
                            epsabs=1e-8, epsrel=1e-6, eps_guard=1e-16, ngq_fallback=160)

    # convertir el dict a matriz 6x6
    P_sph_CMT = assemble_P_voigt_from_P_dict_full(P_sph_CMT_dict)


    f_wetcol = fbvtv_wetcol.copy()
    f_HA = fbvtv_HA.copy()
    f_CMT = C_MT.copy()
    P_sph_voigt =P_sph_CMT.copy()
    #eigvals = np.linalg.eigvals(C0_voigt)
   # print(f"[ic={ic}] eigvals: {eigvals.real}")

    F_C_voigh = C0_voigt.copy()
    P_cyl_voigt = build_P_cyl_col_from_Cvoigh(F_C_voigh, eps=1e-12)
    
    # print(f"[INFO]:[f_wetcol] len={f_wetcol.size}, dtype={f_wetcol.dtype}, ")
    # print(f"[INFO]:[f_HA]len={f_HA.size}, dtype={f_HA.dtype},")
    # print(f"[INFO]:[f_CMT] len={f_CMT.size}, dtype={f_CMT.dtype}, ")
    # print(f"[INFO]:[F_C_voigh] len={F_C_voigh.size}, dtype={F_C_voigh.dtype}")
    # print(f"[INFO]:[P_cyl_voigh] len={P_cyl_voigt.size}, dtype={P_cyl_voigt.dtype}")
    # print(f"[INFO]:[P_sph_voigh] len={P_sph_voigt.size}, dtype={P_sph_voigt.dtype}")
    # # eig_cyl = np.linalg.eigvals(f_CMT - F_C_voigh)
    # # eig_sph = np.linalg.eigvals(C_HA - F_C_voigh)
    # print(f"[DEBUG] ic={ic}, eig(C_m - C0): {np.round(eig_cyl, 3)}")
    C_fibrilla_local[ic, :, :] = MT_two_families_from_fibrilla(C_MT, C0_voigt, P_cyl_voigt, C_HA, P_sph_voigt, f_wetcol, f_HA, rcond=1e-12, reg_eps=1e-12)
   # MT_two_families_from_fibrilla(f_CMT, F_C_voigh, P_cyl_voigh, C_HA, P_sph_, f_wetcol, f_HA)
   # C_fibrilla_local[ic, :, :] = symm6(Ceff_voigt[ic])
                          

# --- chequeo global rápido ---
local_min = np.min(C_fibrilla_local.reshape(n_cells_local, -1))
local_max = np.max(C_fibrilla_local.reshape(n_cells_local, -1))
glob_min = comm.allreduce(local_min, op=MPI.MIN)
glob_max = comm.allreduce(local_max, op=MPI.MAX)
if rank == 0:
    print(f"[INFO] C_fibrilla_local rango global: min={glob_min:.6g}, max={glob_max:.6g}")

# --- Export: ejemplo exportar componente C11 por celda a XDMF para ParaView ---


# 1️⃣ Calcular baricentros de celdas
mesh_coords = mesh.geometry.x
connectivity = mesh.topology.connectivity(mesh.topology.dim, 0)
cell_bary = np.zeros((n_cells, 3))

for ic in range(n_cells):
    node_ids = connectivity.links(ic)
    cell_coords = mesh_coords[node_ids]
    cell_bary[ic] = np.mean(cell_coords, axis=0)

# 2️⃣ Tomar el valor escalar/tensorial de cada celda
values = C_fibrilla_local[:, 0, 0]  # ejemplo: componente (0,0)

# 3️⃣ Interpolar a nodos
interp = LinearNDInterpolator(cell_bary, values)
node_values = interp(mesh_coords)
node_values = np.nan_to_num(node_values, nan=np.nanmean(node_values))

# Crear función escalar
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
C11 = dolfinx.fem.Function(V, name="C11_Fibrilla")

# Asignar valores a los nodos
C11.x.array[:] = node_values.flatten()

# Guardar para Paraview *** Aca estamos imprimendo nodos.
if rank == 0:
    with XDMFFile(MPI.COMM_WORLD, "./17_11_25/C11_fibrilla.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(C11)
if rank == 0:
    print(f"[OK] C_Fibrilla_ construido. Componente rango global: min={glob_min:.6g}, max={glob_max:.6g}")




# =============================================================================
# =============================================================================
# ------------Etapa -- 3-- 1) Funciones de Carga              -----------------
# =============================================================================
# =============================================================================




# ------------Etapa -- 3-- 1) Datos (ya calculados por celda) -----------------
# Suponemos que:
#  - bvtv_HA_array: shape (n_cells_X,6,6) con np.eye(6)*val_HA por celda
#  - bvtv_ic_array: shape (n_cells_X,6,6) con np.eye(6)*val_ic por celda
# Pueden ser arrays globales (n_cells_global) o locales (n_cells_local).

HA_arr = np.asarray(bvtv_HA_array, dtype=np.float64)
ic_arr = np.asarray(bvtv_ic_array, dtype=np.float64)
# ----------------- 7) Prealocar C_foam_cell local y bucle MT -----------------
C_Extra_foam_local = np.zeros((n_cells_local, 6, 6), dtype=np.float64)

# --- bucle local: llamar MT_two_families por celda ---
for ic in range(n_cells_local):
   # --- extraer fracciones escalares desde la matriz 6x6 ---
    # opción A: usar trace/6 (promedio diagonal)
    fbvtv_ic = bvtv_ic_array[ic, :, :]  # matriz 6x6, no float
    fbvtv_HA = bvtv_HA_array[ic, :, :]
    f_total = fbvtv_ic + fbvtv_HA

    C_Ef= C_fibrilla_local[ic, :, :]
    C0_voigh = C_Ef*f_total
    
    P_sph_CMT_dict = compute_P_sph_from_C0_adaptive_quad(C0_voigt,
                            epsabs=1e-8, epsrel=1e-6, eps_guard=1e-16, ngq_fallback=160)

    # convertir el dict a matriz 6x6
    P_sph_ef = assemble_P_voigt_from_P_dict_full(P_sph_CMT_dict)



    f_ic = fbvtv_ic.copy()
    f_HA = fbvtv_HA.copy()
    f_CMT = C_MT.copy()
    P_sph_ =P_sph_voigh.copy()
    #eigvals = np.linalg.eigvals(C_voigh)
    #print(f"[ic={ic}] eigvals: {eigvals.real}")

    # F_C_voigh = C0_voigh.copy()
    # P_cyl_voigh = build_P_cyl_col_from_Cvoigh(F_C_voigh, eps=1e-12)
    
    # print(f"[INFO]:[f_ic] len={f_ic.size}, dtype={f_ic.dtype}, ")
    # print(f"[INFO]:[f_HA]len={f_HA.size}, dtype={f_HA.dtype},")
    # print(f"[INFO]:[f_CMT] len={f_CMT.size}, dtype={f_CMT.dtype}, ")
    # print(f"[INFO]:[F_C_voigh] len={F_C_voigh.size}, dtype={F_C_voigh.dtype}")
    # print(f"[INFO]:[P_cyl_voigh] len={P_cyl_voigh.size}, dtype={P_cyl_voigh.dtype}")
    # print(f"[INFO]:[P_sph_] len={P_sph_.size}, dtype={P_sph_voigh.dtype}")
    # eig_cyl = np.linalg.eigvals(f_CMT - F_C_voigh)
    # eig_sph = np.linalg.eigvals(C_HA - F_C_voigh)
    # print("[DEBUG] eig(C_m - C0):", np.round(eig_cyl, 3))
    # print("[DEBUG] eig(C_HA - C0):", np.round(eig_sph, 3))
    
    C_Extra_foam_local[ic, :, :] =MT_two_families_from_Ef(C_im, C_Ef, P_sph_ef, C_HA, f_ic, f_HA, rcond=1e-12, reg_eps=1e-12)
    #MT_two_families_from_Ef(f_CMT, F_C_voigh, P_cyl_voigh, C_HA, P_sph_, f_ic, f_HA)
   # C_fibrilla_local[ic, :, :] = symm6(Ceff_voigt[ic])
                          

# --- chequeo global rápido ---
local_min = np.min(C_Extra_foam_local.reshape(n_cells_local, -1))
local_max = np.max(C_Extra_foam_local.reshape(n_cells_local, -1))
glob_min = comm.allreduce(local_min, op=MPI.MIN)
glob_max = comm.allreduce(local_max, op=MPI.MAX)
if rank == 0:
    print(f"[INFO] C_Extra_foam_local rango global: min={glob_min:.6g}, max={glob_max:.6g}")

# --- Export: ejemplo exportar componente C11 por celda a XDMF para ParaView ---


# 1️⃣ Calcular baricentros de celdas
mesh_coords = mesh.geometry.x
connectivity = mesh.topology.connectivity(mesh.topology.dim, 0)
cell_bary = np.zeros((n_cells, 3))

for ic in range(n_cells):
    node_ids = connectivity.links(ic)
    cell_coords = mesh_coords[node_ids]
    cell_bary[ic] = np.mean(cell_coords, axis=0)

# 2️⃣ Tomar el valor escalar/tensorial de cada celda
values = C_Extra_foam_local[:, 0, 0]  # ejemplo: componente (0,0)

# 3️⃣ Interpolar a nodos
interp = LinearNDInterpolator(cell_bary, values)
node_values = interp(mesh_coords)
node_values = np.nan_to_num(node_values, nan=np.nanmean(node_values))

# Crear función escalar
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
C11_foam = dolfinx.fem.Function(V, name="C11_extra_foam")

# Asignar valores a los nodos
C11_foam.x.array[:] = node_values.flatten()

# Guardar para Paraview *** Aca estamos imprimendo nodos.
if rank == 0:
    with XDMFFile(MPI.COMM_WORLD, "./17_11_25/C11_extra_foam.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(C11_foam)

# ----------------- 9) Resumen global (min/max) -----------------
local_min = np.min(C_Extra_foam_local.reshape(n_cells_local, -1))
local_max = np.max(C_Extra_foam_local.reshape(n_cells_local, -1))
glob_min = comm.allreduce(local_min, op=MPI.MIN)
glob_max = comm.allreduce(local_max, op=MPI.MAX)
if rank == 0:
    print(f"\n[Etapa3] rango global C_foam_cell: min={glob_min:.6g}, max={glob_max:.6g}")


# # # -------------------------------------------------------
# # # Etapa 4 — Ultra (mezcla de dominio: fibrilla + espuma, cilindros)
# # # -------------------------------------------------------
# # # -------------------------------------------------------
# # # -------------------------------------------------------
# # # -------------------------------------------------------
# # # -------------------------------------------------------

f_fib = np.asarray(bvtv_fib_array , dtype=np.float64)
f_ef = np.asarray(bvtv_ef_array, dtype=np.float64)


# ----------------- 7) Prealocar C_Extra_cell_local y bucle MT -----------------
C_Ultra_cell_local = np.zeros((n_cells_local, 6, 6), dtype=np.float64)



# --- bucle local: llamar MT_two_families por celda ---
for ic in range(n_cells_local):
   # --- extraer fracciones escalares desde la matriz 6x6 ---
    # opción A: usar trace/6 (promedio diagonal)
    f_fib = bvtv_fib_array[ic, :, :]  # matriz 6x6, no float
    f_ef = bvtv_ef_array[ic, :, :]

    C_fib= C_fibrilla_local[ic, :, :]
    C_Ef = C_Extra_foam_local[ic,:,:]
    
    F_C_voigh = C_Ef.copy()
    P_cyl_ef = build_P_cyl_col_from_Cvoigh(F_C_voigh, eps=1e-12)
    # convertir el dict a matriz 6x6

    f_ic = fbvtv_ic.copy()
    f_HA = fbvtv_HA.copy()
    f_CMT = C_MT.copy()
    P_sph_ =P_sph_voigh.copy()
    C_fib = C_Ef
    #eigvals = np.linalg.eigvals(C_voigh)
    print(f"[ultra-ic={ic}]")    
    C_Ultra_cell_local[ic, :, :] = MT_two_families_from_ExtraC(C_im, C_Ef, P_cyl_ef, C_fib,
                                f_fib, f_ef, rcond=1e-12, reg_eps=1e-12)
    

# --- chequeo global rápido ---
local_min = np.min(C_Ultra_cell_local.reshape(n_cells_local, -1))
local_max = np.max(C_Ultra_cell_local.reshape(n_cells_local, -1))
glob_min = comm.allreduce(local_min, op=MPI.MIN)
glob_max = comm.allreduce(local_max, op=MPI.MAX)
if rank == 0:
    print(f"[INFO] C_Ultra_cell_local rango global: min={glob_min:.6g}, max={glob_max:.6g}")

# --- Export: ejemplo exportar componente C11 por celda a XDMF para ParaView ---


# 1️⃣ Calcular baricentros de celdas
mesh_coords = mesh.geometry.x
connectivity = mesh.topology.connectivity(mesh.topology.dim, 0)
cell_bary = np.zeros((n_cells, 3))

for ic in range(n_cells):
    node_ids = connectivity.links(ic)
    cell_coords = mesh_coords[node_ids]
    cell_bary[ic] = np.mean(cell_coords, axis=0)

# 2️⃣ Tomar el valor escalar/tensorial de cada celda
values = C_Extra_foam_local[:, 0, 0]  # ejemplo: componente (0,0)

# 3️⃣ Interpolar a nodos
interp = LinearNDInterpolator(cell_bary, values)
node_values = interp(mesh_coords)
node_values = np.nan_to_num(node_values, nan=np.nanmean(node_values))

# Crear función escalar
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
C11_foam = dolfinx.fem.Function(V, name="C11_extra_cell")

# Asignar valores a los nodos
C11_foam.x.array[:] = node_values.flatten()

# Guardar para Paraview *** Aca estamos imprimendo nodos.
if rank == 0:
    with XDMFFile(MPI.COMM_WORLD, "./17_11_25/C11_Ultra_cell.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(C11_foam)

# ----------------- 9) Resumen global (min/max) -----------------
local_min = np.min(C_Extra_foam_local.reshape(n_cells_local, -1))
local_max = np.max(C_Extra_foam_local.reshape(n_cells_local, -1))
glob_min = comm.allreduce(local_min, op=MPI.MIN)
glob_max = comm.allreduce(local_max, op=MPI.MAX)
if rank == 0:
    print(f"\n[Etapa4] rango global C_Ultra_cell_local: min={glob_min:.6g}, max={glob_max:.6g}")




# # -------------------------------------------------------
# #  Etapa 5 — Lagunas (agua esférica sobre C_ultra) + propiedades
# # -------------------------------------------------------



f_phi=np.zeros((n_cells_local, 6, 6), dtype=np.float64)

# ----------------- 7) Prealocar C_Extra_cell_local y bucle MT -----------------
C_extra_vasc_cell_local = np.zeros((n_cells_local, 6, 6), dtype=np.float64)



# --- bucle local: llamar MT_two_families por celda ---
for ic in range(n_cells_local):
   # --- extraer fracciones escalares desde la matriz 6x6 ---
    # opción A: usar trace/6 (promedio diagonal)
    f_phi = bvtv_phi_array[ic,:,:]
    C_ultra= C_Extra_foam_local[ic, :, :]*f_phi     
    P_sph_CMT_dict = compute_P_sph_from_C0_adaptive_quad(C_ultra,
                             epsabs=1e-8, epsrel=1e-6, eps_guard=1e-16, ngq_fallback=160)

     # convertir el dict a matriz 6x6
    P_sph_CMT = assemble_P_voigt_from_P_dict_full(P_sph_CMT_dict)

    phi_ = f_phi.copy()
    P_sph_ultra =P_sph_CMT.copy()    
    # convertir el dict a matriz 6x6

    f_ic = fbvtv_ic.copy()
    f_HA = fbvtv_HA.copy()
    f_CMT = C_MT.copy()
    P_sph_ =P_sph_voigh.copy()
    C_ic = C_im.copy()
    
    #eigvals = np.linalg.eigvals(C_voigh)
    print(f"[ultra-ic={ic}]")    
    C_extra_vasc_cell_local[ic, :, :] = MT_two_families_from_ExtraC(C_ic, C_ultra, P_sph_ultra, phi_, rcond=1e-12, reg_eps=1e-12)
    

# --- chequeo global rápido ---
local_min = np.min(C_extra_vasc_cell_local.reshape(n_cells_local, -1))
local_max = np.max(C_extra_vasc_cell_local.reshape(n_cells_local, -1))
glob_min = comm.allreduce(local_min, op=MPI.MIN)
glob_max = comm.allreduce(local_max, op=MPI.MAX)
if rank == 0:
    print(f"[INFO] C_Ultra_cell_local rango global: min={glob_min:.6g}, max={glob_max:.6g}")

# --- Export: ejemplo exportar componente C11 por celda a XDMF para ParaView ---


# 1️⃣ Calcular baricentros de celdas
mesh_coords = mesh.geometry.x
connectivity = mesh.topology.connectivity(mesh.topology.dim, 0)
cell_bary = np.zeros((n_cells, 3))

for ic in range(n_cells):
    node_ids = connectivity.links(ic)
    cell_coords = mesh_coords[node_ids]
    cell_bary[ic] = np.mean(cell_coords, axis=0)

# 2️⃣ Tomar el valor escalar/tensorial de cada celda
values = C_Ultra_cell_local[:, 0, 0]  # ejemplo: componente (0,0)

# 3️⃣ Interpolar a nodos
interp = LinearNDInterpolator(cell_bary, values)
node_values = interp(mesh_coords)
node_values = np.nan_to_num(node_values, nan=np.nanmean(node_values))

# Crear función escalar
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
C11_extra_vasc = dolfinx.fem.Function(V, name="C11_extra_cell")

# Asignar valores a los nodos
C11_extra_vasc.x.array[:] = node_values.flatten()

# Guardar para Paraview *** Aca estamos imprimendo nodos.
if rank == 0:
    with XDMFFile(MPI.COMM_WORLD, "./17_11_25/C11_extra_vasc.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(C11_extra_vasc)

# ----------------- 9) Resumen global (min/max) -----------------
local_min = np.min(C_extra_vasc_cell_local.reshape(n_cells_local, -1))
local_max = np.max(C_extra_vasc_cell_local.reshape(n_cells_local, -1))
glob_min = comm.allreduce(local_min, op=MPI.MIN)
glob_max = comm.allreduce(local_max, op=MPI.MAX)
if rank == 0:
    print(f"\n[Etapa5] rango global C_Ultra_cell_local: min={glob_min:.6g}, max={glob_max:.6g}")



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ---------------------------------------------------------------------
# Parámetros de regularización/inversión
EPS_EIG_CLIP = 1e-8   # clip para eigenvals pequeños
EPS_INV = 1e-12       # valor usado si todo falla
# ---------------------------------------------------------------------

def safe_inverse_6x6(C, eps_clip=EPS_EIG_CLIP):
    """
    Inversión estable de matriz 6x6 simétrica.
    - Si C es SPD suficiente: inv directa.
    - Si no: hacemos eigen-decomposition y revertimos con clip de autovalores.
    Devuelve S = inv(C).
    """
    # fuerza simetría numérica
    Csym = 0.5 * (C + C.T)
    try:
        # intento directo (rápido para casos bien condicionados)
        S = np.linalg.inv(Csym)
        # si hay NaNs o inf, forzamos fallback
        if not np.isfinite(S).all():
            raise np.linalg.LinAlgError("inv returned non-finite")
        return 0.5*(S + S.T)
    except np.linalg.LinAlgError:
        # fallback estable por valores propios
        w, v = np.linalg.eigh(Csym)
        # clip autovalores pequeños/negativos
        w_clipped = np.clip(w, eps_clip, None)
        S = (v @ np.diag(1.0 / w_clipped) @ v.T)
        return 0.5*(S + S.T)

def moduli_from_C_voigt(C_voigt):
    """
    Entrada: C_voigt 6x6 (rigidez) en notación de Voigt con orden [11,22,33,23,13,12]
    Salida: diccionario con E1,E2,E3, G12,G13,G23 y nu_ij (nu12, nu21,...)
    """
    # seguridad: shape
    assert C_voigt.shape == (6,6)
    # invertir con método estable
    S = safe_inverse_6x6(C_voigt)

    # Youngs (E_i = 1 / S_ii)
    E1 = 1.0 / S[0,0] if abs(S[0,0])>0 else np.nan
    E2 = 1.0 / S[1,1] if abs(S[1,1])>0 else np.nan
    E3 = 1.0 / S[2,2] if abs(S[2,2])>0 else np.nan

    # Shear moduli (voigt indices: 3->23, 4->13, 5->12)
    G23 = 1.0 / S[3,3] if abs(S[3,3])>0 else np.nan
    G13 = 1.0 / S[4,4] if abs(S[4,4])>0 else np.nan
    G12 = 1.0 / S[5,5] if abs(S[5,5])>0 else np.nan

    # Poisson ratios: nu_ij = -S_ij / S_ii
    nu12 = -S[0,1] / S[0,0] if abs(S[0,0])>0 else np.nan
    nu13 = -S[0,2] / S[0,0] if abs(S[0,0])>0 else np.nan
    nu21 = -S[1,0] / S[1,1] if abs(S[1,1])>0 else np.nan
    nu23 = -S[1,2] / S[1,1] if abs(S[1,1])>0 else np.nan
    nu31 = -S[2,0] / S[2,2] if abs(S[2,2])>0 else np.nan
    nu32 = -S[2,1] / S[2,2] if abs(S[2,2])>0 else np.nan

    return {
        "E1": E1, "E2": E2, "E3": E3,
        "G12": G12, "G13": G13, "G23": G23,
        "nu12": nu12, "nu13": nu13, "nu21": nu21,
        "nu23": nu23, "nu31": nu31, "nu32": nu32,
        "S": S  # si quieres inspeccionar la matriz de cumplimiento
    }

# ---------------------------------------------------------------------
# Variables de entrada esperadas:
# - C_cells: array local con forma (n_cells_local, 6, 6) con rigidez por celda (Voigt)
# - mesh: dolfinx mesh
# - n_cells (global)
# - n_cells_local (local)
# - comm, rank ya definidos arriba
#
# Ajusta el nombre C_cells si tu variable se llama distinto (por ejemplo C_fibrilla_local).
# ---------------------------------------------------------------------

# -- ejemplo: si tu rigidez ya está en C_fibrilla_local con shape (n_cells_local,6,6)
C_cells = C_Ultra_cell_local   # -> reemplaza si tu variable tiene otro nombre

# arrays locales para guardar resultados por celda (local chunk)
E1_loc = np.zeros(n_cells_local, dtype=float)
E2_loc = np.zeros(n_cells_local, dtype=float)
E3_loc = np.zeros(n_cells_local, dtype=float)
G12_loc = np.zeros(n_cells_local, dtype=float)
G13_loc = np.zeros(n_cells_local, dtype=float)
G23_loc = np.zeros(n_cells_local, dtype=float)
nu12_loc = np.zeros(n_cells_local, dtype=float)
nu13_loc = np.zeros(n_cells_local, dtype=float)
nu21_loc = np.zeros(n_cells_local, dtype=float)
nu23_loc = np.zeros(n_cells_local, dtype=float)
nu31_loc = np.zeros(n_cells_local, dtype=float)
nu32_loc = np.zeros(n_cells_local, dtype=float)

# Optionally guard NaNs with fill value
FILL_NAN = np.nan

# Loop local
for ic in range(n_cells_local):
    Cc = C_cells[ic, :, :].astype(float)
    # small check for symmetry
    Cc = 0.5*(Cc + Cc.T)
    try:
        mods = moduli_from_C_voigt(Cc)
    except Exception as e:
        # si algo falla, rellenar NaNs y continuar
        if rank == 0:
            print(f"[WARN] fallo al procesar celda local {ic}: {e}")
        mods = {k: np.nan for k in ["E1","E2","E3","G12","G13","G23",
                                    "nu12","nu13","nu21","nu23","nu31","nu32","S"]}

    E1_loc[ic] = mods["E1"] if np.isfinite(mods["E1"]) else FILL_NAN
    E2_loc[ic] = mods["E2"] if np.isfinite(mods["E2"]) else FILL_NAN
    E3_loc[ic] = mods["E3"] if np.isfinite(mods["E3"]) else FILL_NAN
    G12_loc[ic] = mods["G12"] if np.isfinite(mods["G12"]) else FILL_NAN
    G13_loc[ic] = mods["G13"] if np.isfinite(mods["G13"]) else FILL_NAN
    G23_loc[ic] = mods["G23"] if np.isfinite(mods["G23"]) else FILL_NAN
    nu12_loc[ic] = mods["nu12"] if np.isfinite(mods["nu12"]) else FILL_NAN
    nu13_loc[ic] = mods["nu13"] if np.isfinite(mods["nu13"]) else FILL_NAN
    nu21_loc[ic] = mods["nu21"] if np.isfinite(mods["nu21"]) else FILL_NAN
    nu23_loc[ic] = mods["nu23"] if np.isfinite(mods["nu23"]) else FILL_NAN
    nu31_loc[ic] = mods["nu31"] if np.isfinite(mods["nu31"]) else FILL_NAN
    nu32_loc[ic] = mods["nu32"] if np.isfinite(mods["nu32"]) else FILL_NAN

# --- reducción global (opcional) para summary ---
glob_min_E1 = comm.allreduce(np.nanmin(np.where(np.isfinite(E1_loc), E1_loc, np.inf)), op=MPI.MIN)
glob_max_E1 = comm.allreduce(np.nanmax(np.where(np.isfinite(E1_loc), E1_loc, -np.inf)), op=MPI.MAX)

if rank == 0:
    print(f"[INFO] E1 rango global: min={glob_min_E1:.6g}, max={glob_max_E1:.6g}")

# ---------------------------------------------------------------------
# Export a XDMF: interpolar por celda -> nodos (igual que tu flujo anterior)
# Crear dict de campos a exportar
fields = {
    "E1": E1_loc, "E2": E2_loc, "E3": E3_loc,
    "G12": G12_loc, "G13": G13_loc, "G23": G23_loc,
    "nu12": nu12_loc, "nu13": nu13_loc, "nu21": nu21_loc,
    "nu23": nu23_loc, "nu31": nu31_loc, "nu32": nu32_loc
}

# 1) calcular baricentros (global n_cells)
mesh_coords = mesh.geometry.x
connectivity = mesh.topology.connectivity(mesh.topology.dim, 0)
# n_cells (global) asumido ya definido
cell_bary = np.zeros((n_cells, 3), dtype=float)

# Necesitamos llenar cell_bary en todos los ranks. Si el mesh está repartido, debemos
# computar baricentros globalmente - aquí asumimos que `connectivity.links(ic)` funciona globalmente.
# Si tu mesh está particionado, usa el mismo método que tenías (debería funcionar).
for ic in range(n_cells):
    node_ids = connectivity.links(ic)
    # en meshes particionados, algunos índices pueden no existir localmente; en tu script previo
    # esto ya funcionó, así que dejamos igual.
    cell_coords = mesh_coords[node_ids]
    cell_bary[ic] = np.mean(cell_coords, axis=0)

# Ahora, cada campo: values por celda (global). 
# Necesitamos acumular arrays locales al arreglo global ordering por celdas.
# Si tu arreglo fields[...] está en el orden local (n_cells_local) y tienes información para mapear
# a índice global, adapta aquí. Aquí asumimos que las celdas locales están alineadas con
# un bloque contiguo en el orden global (si no, reemplaza con el mapeo de índices globales).
# Para seguridad, construimos un vector global por comunicación (gather).
# -> Simplificamos: cada proceso envía su bloque y el root concatena (esto asume particionado por bloques).

# Recolectar los arrays locales en rank 0
all_fields_global = {}
for name, arr_loc in fields.items():
    # gather lengths y datos en root
    gathered = comm.gather(arr_loc, root=0)
    if rank == 0:
        arr_global = np.concatenate(gathered)
        all_fields_global[name] = arr_global
    else:
        # los no-root no guardan
        pass

# En rank 0 hacemos la interpolación y export XDMF
if rank == 0:
    # sanity checks
    assert len(cell_bary) == n_cells
    # crear espacio y función para nodos
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    # folder de salida
    out_xdmf = "./17_11_25/results_poisson_moduli.xdmf"
    with XDMFFile(MPI.COMM_WORLD, out_xdmf, "w") as xdmf:
        xdmf.write_mesh(mesh)
        for name, values_global in all_fields_global.items():
            # si hay NaNs, reemplazamos por la media (evita errores en interpolación)
            values_clean = np.copy(values_global)
            if np.isnan(values_clean).all():
                print(f"[WARN] campo {name} todo NaN, se salta.")
                continue
            nanmask = np.isnan(values_clean)
            if np.any(nanmask):
                mean_val = np.nanmean(values_clean)
                values_clean[nanmask] = mean_val

            # Interpolar a nodos (usa scipy LinearNDInterpolator)
            interp = LinearNDInterpolator(cell_bary, values_clean)
            node_values = interp(mesh_coords)
            # fallback si quedan NaN: reemplazar por media de nodos (o celda)
            node_values = np.nan_to_num(node_values, nan=np.nanmean(node_values))

            f = dolfinx.fem.Function(V, name=name)
            # Asegurar que el tamaño coincida
            if f.x.array.size != node_values.size:
                raise RuntimeError(f"mismatch nodal size for field {name}: {f.x.array.size} vs {node_values.size}")
            f.x.array[:] = node_values.flatten()
            xdmf.write_function(f)
            print(f"[INFO] exportado campo {name} a XDMF.")

    print(f"[DONE] XDMF guardado en: {out_xdmf}")

# FIN