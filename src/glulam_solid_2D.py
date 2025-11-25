from __future__ import print_function, absolute_import, division    #makes KratosMultiphysics backward compatible with python 2.6 and 2.7

from config_hbv import (
    GEOMETRY,
    MATERIAL_TIMBER,
    MATERIAL_CONCRETE,
    MATERIAL_KERVE,
    SUPPORT,
    LOADCASE,
)

'''Timber Engineering - KratosMultiphysics'''
# Help for Data_Structure: Kratos/docs/pages/Kratos/For_Users/Crash_course/Data_Structure
# Good example: Kratos/applications/StructuralMechanicsApplication/tests/test_prebuckling_analysis.py

'''Import packages'''
# KratosMultiphysics
import KratosMultiphysics
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.vtk_output_process import VtkOutputProcess
from KratosMultiphysics.StructuralMechanicsApplication.displacement_control_with_direction_process import AssignDisplacementControlProcess
from KratosMultiphysics.assign_vector_variable_process import AssignVectorVariableProcess


from KratosMultiphysics import ConstitutiveLawsApplication

# Stuff
import pandas as pd
import numpy as np

'''Import functions'''
from def_node_element_creator import define_nodes
from def_node_element_creator import define_quadrilateralN4
from def_plot import plot_data_2D

'''Glulam Beam with Solid 2D element'''

'Structure'
# First part of the code consist of def functions.
# Second part of the code consist the application of functions 

'to do'
# Check if load can be apllied to node without element
# Check solver settings and their influence
# Constitutive Law for timber (2D-asymetric stiffness matrix)
# Plastification of Timber on Compressionside
# Model Tension failure
# Monte-Carlo Simulation
#
#

'Input'
l = GEOMETRY["length"]
b = GEOMETRY["width"]
h = GEOMETRY["height"]
n_element_x = GEOMETRY["n_el_x"]
n_element_y = GEOMETRY["n_el_y"]
n_element_z = GEOMETRY["n_el_z"]

print("Länge aus Config", 1)

'Dataframe with nodes and elements'
df_nodes, df_nodes_xyz, df_nodes_number = define_nodes(l,b,h,n_element_x,n_element_y,n_element_z)
df_elements, df_elements_number = define_quadrilateralN4(n_element_x,n_element_y,n_element_z)


####################################################################################################
'First part: Define function for setup the numerical model - Pre-processing'
####################################################################################################

'Model and Modelparts'
def model_modelpart():
    # Create Model
    model = KratosMultiphysics.Model()
    # Create Modelpart "model_part_structure"
    model_part_structure = model.CreateModelPart("model_part_structure")
    model_part_structure.SetBufferSize(2)
    # Create SubModelPart "boundary_condition_support_model_part_structure"
    boundary_condition_support_model_part_structure = model_part_structure.CreateSubModelPart("boundary_condition_support_model_part_structure")
    # Create SubModelPart "boundary_condition_load_model_part_structure"
    boundary_condition_load_model_part_structure = model_part_structure.CreateSubModelPart("boundary_condition_load_model_part_structure")
    # Create SubModelPart "boundary_condition_load_lc1_part_structure"
    boundary_condition_load_lc1_model_part_structure = boundary_condition_load_model_part_structure.CreateSubModelPart("boundary_condition_load_lc1_model_part_structure")
    # Create SubModelPart "boundary_condition_load_lc2_part_structure"
    boundary_condition_load_lc2_model_part_structure = boundary_condition_load_model_part_structure.CreateSubModelPart("boundary_condition_load_lc2_model_part_structure")
    # Create SubModelPart "boundary_condition_load_lc3_part_structure"
    boundary_condition_load_lc3_model_part_structure = boundary_condition_load_model_part_structure.CreateSubModelPart("boundary_condition_load_lc3_model_part_structure")
    return model, model_part_structure, boundary_condition_support_model_part_structure, boundary_condition_load_model_part_structure, boundary_condition_load_lc1_model_part_structure, boundary_condition_load_lc2_model_part_structure, boundary_condition_load_lc3_model_part_structure

'Variables'
def variables(model_part_structure):
    # Define variables to ModelPart
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.REACTION)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.LOAD_FACTOR)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.PRESCRIBED_DISPLACEMENT)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.POSITIVE_FACE_PRESSURE)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.NEGATIVE_FACE_PRESSURE)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.POINT_LOAD)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.LINE_LOAD)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.SURFACE_LOAD)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.VOLUME_ACCELERATION)

def material(model_part_structure):
    """Definiert die Material-Properties für Holz, Beton, Kerve.

    Aktuell nutzen die Elemente noch NUR Properties[1] (Holz).
    Beton und Kerve sind vorbereitet fürs spätere HBV-Modell.
    """

    # --- Holz (Properties[1]) ---
    props_timber = model_part_structure.GetProperties()[1]
    props_timber.SetValue(KratosMultiphysics.DENSITY,       MATERIAL_TIMBER["density"])
    props_timber.SetValue(KratosMultiphysics.YOUNG_MODULUS, MATERIAL_TIMBER["E"])
    props_timber.SetValue(KratosMultiphysics.SHEAR_MODULUS, MATERIAL_TIMBER["G"])
    props_timber.SetValue(KratosMultiphysics.POISSON_RATIO, MATERIAL_TIMBER["nu"])
    props_timber.SetValue(KratosMultiphysics.THICKNESS,     MATERIAL_TIMBER["thickness"])

    timber_law = KratosMultiphysics.StructuralMechanicsApplication.LinearElasticPlaneStrain2DLaw()
    props_timber.SetValue(KratosMultiphysics.CONSTITUTIVE_LAW, timber_law)

    # --- Beton (Properties[2]) ---
    props_concrete = model_part_structure.GetProperties()[2]
    props_concrete.SetValue(KratosMultiphysics.DENSITY,       MATERIAL_CONCRETE["density"])
    props_concrete.SetValue(KratosMultiphysics.YOUNG_MODULUS, MATERIAL_CONCRETE["E"])
    props_concrete.SetValue(KratosMultiphysics.SHEAR_MODULUS, MATERIAL_CONCRETE["G"])
    props_concrete.SetValue(KratosMultiphysics.POISSON_RATIO, MATERIAL_CONCRETE["nu"])
    props_concrete.SetValue(KratosMultiphysics.THICKNESS,     MATERIAL_CONCRETE["thickness"])

    concrete_law = KratosMultiphysics.StructuralMechanicsApplication.LinearElasticPlaneStrain2DLaw()
    props_concrete.SetValue(KratosMultiphysics.CONSTITUTIVE_LAW, concrete_law)

    # --- Kerve (Properties[3]) ---
    props_kerve = model_part_structure.GetProperties()[3]
    props_kerve.SetValue(KratosMultiphysics.DENSITY,       MATERIAL_KERVE["density"])
    props_kerve.SetValue(KratosMultiphysics.YOUNG_MODULUS, MATERIAL_KERVE["E"])
    props_kerve.SetValue(KratosMultiphysics.SHEAR_MODULUS, MATERIAL_KERVE["G"])
    props_kerve.SetValue(KratosMultiphysics.POISSON_RATIO, MATERIAL_KERVE["nu"])
    props_kerve.SetValue(KratosMultiphysics.THICKNESS,     MATERIAL_KERVE["thickness"])

    kerve_law = KratosMultiphysics.StructuralMechanicsApplication.LinearElasticPlaneStrain2DLaw()
    props_kerve.SetValue(KratosMultiphysics.CONSTITUTIVE_LAW, kerve_law)

    # Plastification Variant 1 (not working)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.DENSITY,0)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.YOUNG_MODULUS,14400)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.SHEAR_MODULUS,900)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.POISSON_RATIO,0.3)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.THICKNESS,100.0)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.SOFTENING_TYPE,0)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.HARDENING_CURVE,3)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.YIELD_STRESS_TENSION,40.0)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.YIELD_STRESS_COMPRESSION,30)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.FRACTURE_ENERGY ,10000000000.0)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.FRACTURE_ENERGY_COMPRESSION ,1000000000)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.STRESS_DAMAGE_CURVE ,[-40])
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.STRAIN_DAMAGE_CURVE ,[5])
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.DAMAGE_TENSION ,0.5)
    
    # constitutive_law = KratosMultiphysics.KratosGlobals.GetConstitutiveLaw("SmallStrainDplusDminusDamageRankineRankine2D").Clone()
    # constitutive_law = KratosMultiphysics.KratosGlobals.GetConstitutiveLaw("SmallStrainDplusDminusDamageVonMisesVonMises2D").Clone()

def classify_elements_with_kerve(df_nodes, df_elements):
    """Gibt df_elements mit Spalte 'region' zurück:
    'TIMBER', 'CONCRETE' oder 'KERVE' (Beton in der Kerve).
    """

    d_n = GEOMETRY["kerven_tiefe_dn"]
    l_n = GEOMETRY["kerven_laenge_ln"]
    l_v = GEOMETRY["vorholz_laenge_lv"]

    x_start = l_v
    x_end   = l_v + l_n
    y_top   = 0.0        # Fuge Beton/Holz
    y_bot   = -d_n       # Unterkante Kerve (im Holz)

    # Nachschlage-Tabelle: Node-ID -> (x, y)
    node_pos = df_nodes.set_index("node")[["x", "y"]]

    regions = []

    for _, el in df_elements.iterrows():
        node_ids = [el["n_0"], el["n_1"], el["n_2"], el["n_3"]]
        coords = node_pos.loc[node_ids]

        x_c = float(coords["x"].mean())
        y_c = float(coords["y"].mean())

        # Oberhalb Fuge -> Betonplatte
        if y_c > y_top:
            regions.append("CONCRETE")
            continue

        # Innerhalb Kervrechteck -> Kerve-Beton
        if (x_start <= x_c <= x_end) and (y_bot <= y_c <= y_top):
            regions.append("KERVE")
            continue

        # Rest -> Holz
        regions.append("TIMBER")

    df_out = df_elements.copy()
    df_out["region"] = regions
    return df_out

df_elements = classify_elements_with_kerve(df_nodes, df_elements)

'Geometry (Nodes and Elements) and DOFs'
def geometry_DOF(model_part_structure, df_nodes, df_elements):
    # --- 1. Nodes im ModelPart anlegen ---
    for i in range(df_nodes.shape[0]):
        model_part_structure.CreateNewNode(
            int(df_nodes["node"][i]),
            df_nodes["x"][i],
            df_nodes["y"][i],
            df_nodes["z"][i],
        )

    # --- 2. Material-SubModelParts anlegen ---
    mp_timber   = model_part_structure.CreateSubModelPart("TIMBER")
    mp_concrete = model_part_structure.CreateSubModelPart("CONCRETE")
    mp_kerve    = model_part_structure.CreateSubModelPart("KERVE")

    # Sets zum Sammeln der Knoten je Materialbereich
    nodes_timber   = set()
    nodes_concrete = set()
    nodes_kerve    = set()

    # --- 3. Elemente anlegen, Properties nach 'region' wählen ---
    for _, row in df_elements.iterrows():
        e_id     = int(row["element"])
        node_ids = [row["n_0"], row["n_1"], row["n_2"], row["n_3"]]
        region   = row.get("region", "TIMBER")  # falls Spalte fehlt → Holz

        if region == "TIMBER":
            props = model_part_structure.GetProperties()[1]   # Holz
            submp = mp_timber
            nodes_timber.update(node_ids)

        elif region == "CONCRETE":
            props = model_part_structure.GetProperties()[2]   # Beton
            submp = mp_concrete
            nodes_concrete.update(node_ids)

        elif region == "KERVE":
            # Kerve ist mit Beton gefüllt → gleiche Properties wie Beton
            props = model_part_structure.GetProperties()[2]   # Beton
            submp = mp_kerve
            nodes_kerve.update(node_ids)

        else:
            # Fallback: behandle unbekannte Region als Holz
            props = model_part_structure.GetProperties()[1]
            submp = mp_timber
            nodes_timber.update(node_ids)

        new_el = model_part_structure.CreateNewElement(
            "SmallDisplacementElement2D4N",
            e_id,
            node_ids,
            props,
        )
        submp.AddElement(new_el.Id)

    # --- 4. Knoten den SubModelParts zuordnen ---
    if nodes_timber:
        mp_timber.AddNodes(list(nodes_timber))
    if nodes_concrete:
        mp_concrete.AddNodes(list(nodes_concrete))
    if nodes_kerve:
        mp_kerve.AddNodes(list(nodes_kerve))

    # --- 5. DOFs definieren (wie bisher auf dem Haupt-ModelPart) ---
    KratosMultiphysics.VariableUtils().AddDof(
        KratosMultiphysics.DISPLACEMENT_X,
        KratosMultiphysics.REACTION_X,
        model_part_structure,
    )
    KratosMultiphysics.VariableUtils().AddDof(
        KratosMultiphysics.DISPLACEMENT_Y,
        KratosMultiphysics.REACTION_Y,
        model_part_structure,
    )
    KratosMultiphysics.VariableUtils().AddDof(
        KratosMultiphysics.DISPLACEMENT_Z,
        KratosMultiphysics.REACTION_Z,
        model_part_structure,
    )
    KratosMultiphysics.VariableUtils().AddDof(
        KratosMultiphysics.StructuralMechanicsApplication.LOAD_FACTOR,
        KratosMultiphysics.StructuralMechanicsApplication.PRESCRIBED_DISPLACEMENT,
        model_part_structure,
    )

'Boundary Condition - Support'
# Define boundary conditions for supports - single-span beam
def boundary_condition_support_single_span_beam(boundary_condition_support_model_part_structure):
    # Create SubModelPart "boundary_condition_fixed_support_model_part_structure"
    boundary_condition_fixed_support_model_part_structure = boundary_condition_support_model_part_structure.CreateSubModelPart("boundary_condition_fixed_support_model_part_structure")
    # Create SubModelPart "boundary_condition_moveable_support_model_part_structure"
    boundary_condition_moveable_support_model_part_structure = boundary_condition_support_model_part_structure.CreateSubModelPart("boundary_condition_moveable_support_model_part_structure") 
    # Define nodes for supports of single-span beam
    boundary_condition_fixed_support_model_part_structure.AddNodes([1500250350])
    boundary_condition_moveable_support_model_part_structure.AddNodes([1500250350+n_element_x*1000000])
    # Function for applying boundary condtions (fixed support) to nodes of a submodelpart
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_X, True, boundary_condition_fixed_support_model_part_structure.Nodes)
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Y, True, boundary_condition_fixed_support_model_part_structure.Nodes)
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Z, True, boundary_condition_fixed_support_model_part_structure.Nodes)
    # Function for applying boundary condtions (moveable support) to nodes of a submodelpart
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Y, True, boundary_condition_moveable_support_model_part_structure.Nodes)
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Z, True, boundary_condition_moveable_support_model_part_structure.Nodes)

# Define boundary conditions for supports - beam under compression/tension
def boundary_condition_support_beam_compression_tension(boundary_condition_support_model_part_structure):
    # Create SubModelPart "boundary_condition_fixed_support_model_part_structure"
    boundary_condition_fixed_support_model_part_structure = boundary_condition_support_model_part_structure.CreateSubModelPart("boundary_condition_fixed_support_model_part_structure")
    # Define nodes for support of beam under compression/tension
    for i in range(int(n_element_y/2+1)):
        if i == 0:
            boundary_condition_fixed_support_model_part_structure.AddNodes([1500250350])
        else:
            boundary_condition_fixed_support_model_part_structure.AddNodes([int(1500250350+i*1000)])
            boundary_condition_fixed_support_model_part_structure.AddNodes([int(1500250350-i*1000)])
    # Function for applying boundary condtions (fixed support) to nodes of a submodelpart
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_X, True, boundary_condition_fixed_support_model_part_structure.Nodes)
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Y, True, boundary_condition_fixed_support_model_part_structure.Nodes)
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Z, True, boundary_condition_fixed_support_model_part_structure.Nodes)

'Boundary Condition - Load'
# Define boundary conditions for load - loadcase 1 (lc1), 4-point-bending
def boundary_condition_load_lc1(model_part_structure,boundary_condition_load_lc1_model_part_structure,load_application,step,load_X,load_Y,load_Z):
    # Define nodes for loadcase 1 (lc1)
    boundary_condition_load_lc1_model_part_structure.AddNodes([int(1500250350 + n_element_x/3*1000000 - n_element_y/2*1000)])
    boundary_condition_load_lc1_model_part_structure.AddNodes([int(1500250350 + n_element_x*2/3*1000000 - n_element_y/2*1000)])
    # Force controlled - Apply point load to nodes
    if load_application == "force_controlled":
        for node in boundary_condition_load_lc1_model_part_structure.Nodes:
            condition_load_model_part_structure = boundary_condition_load_lc1_model_part_structure.CreateNewCondition("PointLoadCondition2D1N",step*10000000000+node.Id,[node.Id],model_part_structure.GetProperties()[1])
            condition_load_model_part_structure.SetValue(KratosMultiphysics.StructuralMechanicsApplication.POINT_LOAD, [load_X,load_Y,load_Z])
    # Displacement controlled - Apply point load to nodes
    elif load_application == "deformation_controlled":
        for node in boundary_condition_load_lc1_model_part_structure.Nodes:
            condition_load_model_part_structure = boundary_condition_load_lc1_model_part_structure.CreateNewCondition("DisplacementControlCondition3D1N",step*10000000000+node.Id,[node.Id],model_part_structure.GetProperties()[1])
            print(step*10000000000+node.Id)
            condition_load_model_part_structure = AssignDisplacementControlProcess(model,KratosMultiphysics.Parameters(f"""
            {{
            "model_part_name": "model_part_structure.boundary_condition_load_model_part_structure.boundary_condition_load_lc1_model_part_structure",
            "direction"       : "y",
            "point_load_value": 1,
            "prescribed_displacement_value" : "{load_Y}"
            }}
            """))
            condition_load_model_part_structure.ExecuteInitializeSolutionStep()

# Define boundary conditions for load - loadcase 2 (lc2), compression/tension load
def boundary_condition_load_lc2(model_part_structure,boundary_condition_load_lc2_model_part_structure,step,load_X,load_Y,load_Z):
    # Define nodes for loadcase 2 (lc2)
    for i in range(int(n_element_y/2+1)):
        if i == 0:
            boundary_condition_load_lc2_model_part_structure.AddNodes([int(1500250350+n_element_x*1000000)])
        else:
            boundary_condition_load_lc2_model_part_structure.AddNodes([int(1500250350+n_element_x*1000000+i*1000)])
            boundary_condition_load_lc2_model_part_structure.AddNodes([int(1500250350+n_element_x*1000000-i*1000)])
    # Apply point load to nodes
    # for node in boundary_condition_load_lc2_model_part_structure.Nodes:
    #     condition_load_model_part_structure = boundary_condition_load_lc2_model_part_structure.CreateNewCondition("PointLoadCondition2D1N",step*10000000000+node.Id,[node.Id],model_part_structure.GetProperties()[1])
    #     condition_load_model_part_structure.SetValue(KratosMultiphysics.StructuralMechanicsApplication.POINT_LOAD, [load_X,load_Y,load_Z])

    # Displacement controlled - Apply point load to nodes
    for node in boundary_condition_load_lc2_model_part_structure.Nodes:
        condition_load_model_part_structure = boundary_condition_load_lc2_model_part_structure.CreateNewCondition("DisplacementControlCondition3D1N",step*10000000000+node.Id,[node.Id],model_part_structure.GetProperties()[1])
        condition_load_model_part_structure = AssignDisplacementControlProcess(model,KratosMultiphysics.Parameters("""
        {
        "model_part_name": "model_part_structure.boundary_condition_load_model_part_structure.boundary_condition_load_lc2_model_part_structure",
        "direction"       : "x",
        "point_load_value": 1,
        "prescribed_displacement_value" : "-0.25"
        }
        """))
        condition_load_model_part_structure.ExecuteInitializeSolutionStep()

# Define boundary conditions for load - loadcase 3 (lc3), load at single node
def boundary_condition_load_lc3(model_part_structure,boundary_condition_load_lc3_model_part_structure,step,load_X,load_Y,load_Z):
    # Define nodes for loadcase 3 (lc3)
    boundary_condition_load_lc3_model_part_structure.AddNodes([int(1522250350)])
    # Apply point load to node
    for node in boundary_condition_load_lc3_model_part_structure.Nodes:
        condition_load_model_part_structure = boundary_condition_load_lc3_model_part_structure.CreateNewCondition("PointLoadCondition2D1N",step*10000000000+node.Id,[node.Id],model_part_structure.GetProperties()[1])
        condition_load_model_part_structure.SetValue(KratosMultiphysics.StructuralMechanicsApplication.POINT_LOAD, [load_X,load_Y,load_Z])

'Solver stategy'
# Not clear what the single function actual do and how they work -> Testing
# Compare with .py in Kratos/applications/StructuralMechanicsApplication/tests

# Define and build solver
def apply_solver(model_part_structure):
    # Setup the solver
    solve_parameters = KratosMultiphysics.Parameters("""
        {
        "problem_data"     : {
        "problem_name"  : "glulam_solid",
        "parallel_type" : "OpenMP",
        "echo_level"    : 1,
        "start_time"    : 0.0,
        "end_time"      : 1.0
        },
        "solver_settings"  : {
        "time_stepping"                        : {
            "time_step" : 1.0
        },
        "solver_type"                          : "Static",
        "model_part_name"                      : "model_part_structure",
        "domain_size"                          : 2,
        "echo_level"                           : 2,
        "analysis_type"                        : "linear",
        "model_import_settings"                : {
            "input_type"     : "use_input_model_part",
            "input_filename" : "glulam_solid_2D/glulam_solid"
        },
        "displacement_control"                 : true,
        "convergence_criterion"                : "residual_criterion",
        "displacement_relative_tolerance"      : 0.0001,
        "displacement_absolute_tolerance"      : 1e-9,
        "residual_relative_tolerance"          : 0.0001,
        "residual_absolute_tolerance"          : 1e-9,
        "max_iteration"                        : 10,
        "use_old_stiffness_in_first_iteration" : false,
        "rotation_dofs"                        : false,
        "volumetric_strain_dofs"               : false
        }
        }
        """)
 
    # Start simulation
    simulation = StructuralMechanicsAnalysis(model, solve_parameters)
    simulation.Run()

'Output - General'
# General Output in Terminal
# print(model_part_structure)

'Output - VTK'
def apply_output_vtk(model):
    vtk_output_process = VtkOutputProcess(model,KratosMultiphysics.Parameters("""
    {
    "model_part_name"                    : "model_part_structure",
    "output_control_type"                : "step",
    "output_interval"                    : 1,
    "file_format"                        : "ascii",
    "output_precision"                   : 7,
    "output_sub_model_parts"             : false,
    "write_deformed_configuration"       : true,
    "output_path"                        : "vtk_output_glulam_solid_2D",
    "save_output_files_in_folder"        : false,
    "nodal_solution_step_data_variables" : ["DISPLACEMENT","REACTION"],
    "gauss_point_variables_extrapolated_to_nodes": ["PK2_STRESS_VECTOR"],
    "gauss_point_variables_in_elements": ["PK2_STRESS_VECTOR"]  
    }
    """))
    vtk_output_process.PrintOutput()
     
'Output system response quantity (SRQ)'
def apply_output_SRQ(model_part_structure):
    # Deformation in nodes
    list_node_deformation = []
    node: KratosMultiphysics.Node
    for node in model_part_structure.Nodes:       
        # Deformation in historical container
        dict_node_deformation = {}
        dict_node_deformation.update({'node':node.Id, 'deformation_x': node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X), 'deformation_y': node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y), 'deformation_z': node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Z)})
        list_node_deformation.append(dict_node_deformation)  
    df_node_deformation = pd.DataFrame(list_node_deformation)
    # Reaction in nodes
    list_node_reaction = []
    node: KratosMultiphysics.Node
    for node in model_part_structure.Nodes:       
        # Deformation in historical container
        dict_node_reaction = {}
        dict_node_reaction.update({'node':node.Id, 'reaction_x': node.GetSolutionStepValue(KratosMultiphysics.REACTION_X), 'reaction_y': node.GetSolutionStepValue(KratosMultiphysics.REACTION_Y), 'reaction_z': node.GetSolutionStepValue(KratosMultiphysics.REACTION_Z)})
        list_node_reaction.append(dict_node_reaction)  
    df_node_reaction = pd.DataFrame(list_node_reaction)
    # Stess in gauss points
    extrapolation_parameters = KratosMultiphysics.Parameters("""
    {
    "model_part_name"            : "model_part_structure",
    "echo_level"                 : 2,
    "average_variable"           : "NODAL_AREA",
    "area_average"               : true,
    "list_of_variables"          : ["PK2_STRESS_VECTOR"],
    "extrapolate_non_historical" : true
    }
    """)
    # Extrapolation stress to nodes
    integration_values_extrapolation_to_nodes_process = KratosMultiphysics.IntegrationValuesExtrapolationToNodesProcess(model_part_structure,extrapolation_parameters)
    integration_values_extrapolation_to_nodes_process.Execute()
    list_node_PK2_stress = []
    node: KratosMultiphysics.Node
    for node in model_part_structure.Nodes:       
        # Stresses in non-historical container
        dict_node_PK2_stress = {}
        dict_node_PK2_stress.update({'node':node.Id, 'stress_x': node[KratosMultiphysics.PK2_STRESS_VECTOR][0], 'stress_y': node[KratosMultiphysics.PK2_STRESS_VECTOR][1], 'stress_z': node[KratosMultiphysics.PK2_STRESS_VECTOR][2]})
        list_node_PK2_stress.append(dict_node_PK2_stress)  
        # Print stresses in non-historical container
        # print(f"node {node.Id}: {node.Has(KratosMultiphysics.PK2_STRESS_VECTOR)}: {node[KratosMultiphysics.PK2_STRESS_VECTOR]}")
    df_node_PK2_stress = pd.DataFrame(list_node_PK2_stress)
    return df_node_deformation, df_node_reaction, df_node_PK2_stress

####################################################################################################
'Second part: Apply numerical model and start calculation'
####################################################################################################

'Define steps'
step = 0
end_step = LOADCASE["n_steps"]

'Boundary Condition - Support'
boundary_condition_support = SUPPORT["type"]

'Define load'
boundary_condition_load = LOADCASE["case"]
load_application = LOADCASE["application"]
load_max = LOADCASE["max_value"]
load_min = load_max / end_step
load_step = np.linspace(load_min, load_max, end_step)

'Setup Model'
# Model and Modelparts
model, model_part_structure, boundary_condition_support_model_part_structure, boundary_condition_load_model_part_structure, boundary_condition_load_lc1_model_part_structure, boundary_condition_load_lc2_model_part_structure, boundary_condition_load_lc3_model_part_structure = model_modelpart()
# Variables
variables(model_part_structure)
# Material and Constitutive Law'
material(model_part_structure)
# Geometry (Nodes and Elements) and DOFs
geometry_DOF(model_part_structure,df_nodes,df_elements)
# Boundary Condition - Support
if boundary_condition_support == "single_span_beam":
    boundary_condition_support_single_span_beam(boundary_condition_support_model_part_structure)
elif boundary_condition_support == "beam_compression_tension":
    boundary_condition_support_beam_compression_tension(boundary_condition_support_model_part_structure)
'Preparation for output'
# List for output
list_load_deformation_curve = []
'Start calculation'
while step < end_step:
    # Updating step
    step += 1
    # Boundary Condition - Load
    if boundary_condition_load == "lc1":
        boundary_condition_load_lc1(model_part_structure,boundary_condition_load_lc1_model_part_structure,load_application,step,0,load_min*step,0)
    elif boundary_condition_load == "lc2":
        boundary_condition_load_lc2(model_part_structure,boundary_condition_load_lc2_model_part_structure,0,0,-100000/(n_element_y+1),0)
    elif boundary_condition_load == "lc3":
        boundary_condition_load_lc3(model_part_structure,boundary_condition_load_lc3_model_part_structure,0,0,85000,0)
    # Solver stategy
    apply_solver(model_part_structure)
    model_part_structure.ProcessInfo[KratosMultiphysics.STEP] = step
    # Ouput - VTK
    apply_output_vtk(model)
    # Output system response quantity (SRQ)
    df_node_deformation, df_node_reaction, df_node_PK2_stress = apply_output_SRQ(model_part_structure)
    # Output - Load-Deformation-Curve
    if boundary_condition_support == "single_span_beam":
        node_id_mid_span = 1500250350+n_element_x/2*1000000
        node_id_support = 1500250350
        node_deformation = df_node_deformation.loc[df_node_deformation['node'] == node_id_mid_span, 'deformation_y'].values[0]
        node_reaction = df_node_reaction.loc[df_node_reaction['node'] == node_id_support, 'reaction_y'].values[0]
        dict_load_deformation_curve = {}
        if load_application == "deformation_controlled":
            dict_load_deformation_curve.update({'deformation':node_deformation, 'load': node_reaction})
        elif load_application == "force_controlled":
            dict_load_deformation_curve.update({'deformation':node_deformation, 'load': load_step[step-1]})
        list_load_deformation_curve.append(dict_load_deformation_curve)
    # elif boundary_condition_support == "beam_compression_tension":
    #     boundary_condition_support_beam_compression_tension(boundary_condition_support_model_part_structure)

####################################################################################################
'Third part: Post-Processing'
####################################################################################################

'Output - Load-Deformation-Curve'
# df_load_deformation_curve = pd.DataFrame(list_load_deformation_curve)
# plot_data_2D('Load-Deformation-Curve',df_load_deformation_curve['deformation'],df_load_deformation_curve['load'],'Load-Deformation-Curve')

'Output - stress at midspan'
# Nodes at midspan
node_id_mid_span = 1500250350+n_element_x/2*1000000
nodes_id_mid_span = [node_id_mid_span-1000*n_element_y/2 + i*1000 for i in range(n_element_y+1)]
# Stresses at midspan
df_node_midspan_PK2_stress = df_node_PK2_stress[df_node_PK2_stress['node'].isin(nodes_id_mid_span)].copy()
# Stresses at midspan with node position
df_node_midspan_xyz = df_nodes[df_nodes['node'].isin(nodes_id_mid_span)].copy()
df_node_midspan_PK2_stress_xyz = pd.merge(df_node_midspan_PK2_stress,df_node_midspan_xyz, on="node", how='left')
# Plot stress distribution
plot_data_2D('Stress Distribution',df_node_midspan_PK2_stress_xyz['stress_x'],df_node_midspan_PK2_stress_xyz['y'],'s_x')

'Output - stress in elements'
# Output elements for evaluation
elements_id_mid_span = [n_element_y/2*(n_element_x-1) + i for i in range(n_element_y)]
# print(elements_id_mid_span)

