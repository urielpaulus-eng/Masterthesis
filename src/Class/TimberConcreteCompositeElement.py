#import the dictionaries and Packages
from src.Objects.materials import wood, concrete 
from src.Objects.geometry import geometry_tcc
from src.Objects.supports import support_tcc
from src.Objects.loadcases import loadcase_tcc
from src.Objects.solver_settings import solver_tcc
import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication
from src.Mesh.mesh_generator import define_nodes, define_quadrilateralN4
from src.Objects.constitutive_laws import constitutive_laws_tcc
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.vtk_output_process import VtkOutputProcess
import os




class TimberConcreteCompositeElement:     #  (TCC)la clase define las propiedades que debe tener el Elemento
        

    def __init__(self):

        self.wood = None
        self.concrete = None
        self.geometry = None
        self.support =  None
        self.loadcase = None
        self.solver_settings = None
        self.kratos_model = None
        self.structure_mp = None     # mp = ModelPart
        self.support_mp = None
        self.load_mp = None
        self.mesh_nodes_df = None     #df = DataFrame
        self.mesh_elements_df = None
        self.element_name = "SmallDisplacementElement2D4N"
        self.timber_law = None
        self.concrete_law = None
        self.kerf_law = None
        self.use_plasticity_parameters = True  # True to activate plasticity
        self.kratos_solver = None
        self.solver_is_initialized = False
        self.vtk_output_by_region = None





        ####################################################################################################
        'First part: Input of TCC Properties - Pre-processing'
        ####################################################################################################

    def call_wood_properties (self):

        if self.wood is None:
            print("Error: self.wood is None. Select a wood type e.g. 'oak'")
            return
        
        if self.wood not in wood:
            print(f"Error: wood type '{self.wood}' not found. Available: {list(wood.keys())}")
            return


        print(wood[self.wood]["density"]) 
        print(wood[self.wood]["youngs_modulus"])

    def call_concrete_properties(self):

    # Safety checks to avoid cryptic KeyErrors
        if self.concrete is None:
            print("Error: self.concrete is None. Set e.g. tcc_element_1.concrete = 'C25/30'")
            return

        if self.concrete not in concrete:
            print(f"Error: concrete type '{self.concrete}' not found. Available: {list(concrete.keys())}")
            return
           
        print(concrete[self.concrete]["density"])
    

    def validate_materials(self):
         # Validate that required material keys are set and exist in the registries
        if self.wood is None:
            raise ValueError("Material error: 'wood' is not set. Example: obj.wood = 'oak'")

        if self.wood not in wood:
            raise KeyError(f"Material error: wood type '{self.wood}' not found. Available: {list(wood.keys())}")

        if self.concrete is None:
            raise ValueError("Material error: 'concrete' is not set. Example: obj.concrete = 'default'")

        if self.concrete not in concrete:
            raise KeyError(f"Material error: concrete type '{self.concrete}' not found. Available: {list(concrete.keys())}")
        
        required_keys = ["density", "youngs_modulus", "shear_modulus", "poissons_ratio", "thickness"]

        for k in required_keys:
            if k not in wood[self.wood]:
                raise KeyError(f"Material error: missing key '{k}' in wood['{self.wood}']")
            if k not in concrete[self.concrete]:
                raise KeyError(f"Material error: missing key '{k}' in concrete['{self.concrete}']")
        
    
    def validate_geometry(self):
    # Basic geometry validation (dictionary must exist and contain key parameters)
        if self.geometry is None:
            raise ValueError("Geometry error: 'geometry' is not set. Example: obj.geometry = geometry_tcc['default']")
         
        required_keys = [
        "tcc_element_length",
        "cross_section_height",
        "cross_section_width",
        "timber_height",
        "concrete_height",
        "kerf_depth",
        "kerf_length",
        "kerf_forewood_length",
        "kerf_flank_angle_deg",
        "num_mesh_elements_x",
        "num_mesh_elements_y",
        "num_mesh_elements_z",
    ]

        missing = [k for k in required_keys if k not in self.geometry]
        if missing:
            raise KeyError(f"Geometry error: missing keys {missing}. Required: {required_keys}")
        
        # Basic numeric checks
        if self.geometry["tcc_element_length"] <= 0:
            raise ValueError("Geometry error: 'tcc_element_length' must be > 0")

        if self.geometry["cross_section_height"] <= 0:
         raise ValueError("Geometry error: 'cross_section_height' must be > 0")

        if self.geometry["num_mesh_elements_x"] <= 29:
            raise ValueError("Geometry error: 'num_mesh_elements_x' must be >=30 ")

        if self.geometry["num_mesh_elements_y"] <= 3:
            raise ValueError("Geometry error: 'num_mesh_elements_y' must be >= 4")
        
        # Consistency check for layer heights
        total_cross_section_height = self.geometry["timber_height"] + self.geometry["concrete_height"]
        if abs(total_cross_section_height - self.geometry["cross_section_height"]) > 1e-6:
            raise ValueError("Geometry error: timber_height + concrete_height must equal cross_section_height")
        
    def call_geometry_properties(self): 
        #Print the currently selected geometry dictionary (stored in self.geometry)
        self.validate_geometry()
        
        print("Selected geometry:")
        for key, value in self.geometry.items():
            print(f"  {key}: {value}")

    def validate_support(self):
        # Validate that a support dictionary is set and has the required key
        if self.support is None:
            raise ValueError("Support error: 'support' is not set. Example: obj.support = supports['single_span_beam']")

        if "type" not in self.support:
            raise KeyError("Support error: missing key 'type' in support dictionary")
        
    def validate_loadcase(self):
        # Validate that a loadcase dictionary is set and has the required keys
        if self.loadcase is None:
            raise ValueError("Loadcase error: 'loadcase' is not set. Example: obj.loadcase = loadcase_tcc['lc4_uniform_line_load']")

        required_keys = ["type", "application", "max_value", "n_steps"]
        missing = [k for k in required_keys if k not in self.loadcase]
    
        if missing:
            raise KeyError(f"Loadcase error: missing keys {missing}. Required: {required_keys}")

        if self.loadcase["n_steps"] <= 0:
            raise ValueError("Loadcase error: 'n_steps' must be >= 1")
        
        ####################################################################################################
        'Second part: Setup of the numerical model - Pre-processing'
        ####################################################################################################
        
    def validate_solver_settings(self):
        # Validate that solver settings dictionary is set and has the required keys
        if self.solver_settings is None:
            raise ValueError("Solver error: 'solver_settings' is not set. Example: obj.solver_settings = solver_tcc['default_static_nonlinear']")

        required_keys = [
        "solver_type", "analysis_type", "domain_size", "echo_level",
        "time_step", "start_time", "end_time",
        "convergence_criterion",
        "residual_relative_tolerance", "residual_absolute_tolerance",
        "max_iteration"
        ]
        missing = [k for k in required_keys if k not in self.solver_settings]
        if missing:
            raise KeyError(f"Solver error: missing keys {missing}. Required: {required_keys}")

        if self.solver_settings["max_iteration"] <= 0:
            raise ValueError("Solver error: 'max_iteration' must be >= 1")

        if self.solver_settings["time_step"] <= 0:
            raise ValueError("Solver error: 'time_step' must be > 0")
        
    def create_kratos_model(self):
        # Create Kratos Model and main structural ModelPart
        self.kratos_model = KratosMultiphysics.Model()
        self.structure_mp = self.kratos_model.CreateModelPart("model_part_structure")
        self.structure_mp.SetBufferSize(2)

        print("Kratos objects created:")
        print("  model =", type(self.kratos_model))
        print("  model_part name =", self.structure_mp.Name)
        print("  buffer size =", self.structure_mp.GetBufferSize())

    def add_kratos_variables(self):
        # Add nodal solution step variables to the main structural ModelPart
        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)
        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.REACTION)

        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.LOAD_FACTOR)
        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.PRESCRIBED_DISPLACEMENT)

        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.POINT_LOAD)
        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.LINE_LOAD)
        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.SURFACE_LOAD)

        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.VOLUME_ACCELERATION)

        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.POSITIVE_FACE_PRESSURE)
        self.structure_mp.AddNodalSolutionStepVariable(KratosMultiphysics.NEGATIVE_FACE_PRESSURE)


        print("Kratos variables added to structure_mp.")

    def create_submodelparts(self):
        # Create sub model parts for supports and loads
        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        self.support_mp = self.structure_mp.CreateSubModelPart("support")
        self.load_mp = self.structure_mp.CreateSubModelPart("loads")

        print("SubModelParts created:")
        print("  support_mp name =", self.support_mp.Name)
        print("  load_mp name =", self.load_mp.Name)
    
    def build_mesh_tables(self):
        # Build node and element tables from the selected geometry
        self.validate_geometry()

        tcc_element_length = float(self.geometry["tcc_element_length"])
        cross_section_height = float(self.geometry["cross_section_height"])
        cross_section_width = float(self.geometry["cross_section_width"])

        num_mesh_elements_x = int(self.geometry["num_mesh_elements_x"])
        num_mesh_elements_y = int(self.geometry["num_mesh_elements_y"])
        num_mesh_elements_z = int(self.geometry["num_mesh_elements_z"])

        self.mesh_nodes_df, _, _ = define_nodes(        # _ signifies that the return of the function is not used
        tcc_element_length,
        cross_section_height,
        cross_section_width,
        num_mesh_elements_x,
        num_mesh_elements_y,
        num_mesh_elements_z,
    )

        self.mesh_elements_df, _ = define_quadrilateralN4(
        num_mesh_elements_x,
        num_mesh_elements_y,
        num_mesh_elements_z,
    )
    
        print("Mesh tables created:")
        print("  mesh_nodes_df rows =", len(self.mesh_nodes_df))
        print("  mesh_elements_df rows =", len(self.mesh_elements_df))
        print("y_min =", float(self.mesh_nodes_df["y"].min()))
        print("y_max =", float(self.mesh_nodes_df["y"].max()))
        print("height_from_mesh =", float(self.mesh_nodes_df["y"].max() - self.mesh_nodes_df["y"].min()))
        print("height_from_geometry =", float(self.geometry["cross_section_height"]))


    def classify_mesh_elements_regions(self):
        if self.mesh_nodes_df is None or self.mesh_elements_df is None:
            raise RuntimeError("Mesh error: mesh tables are not created. Call build_mesh_tables() first.")

        self.validate_geometry()

        cross_section_height = float(self.geometry["cross_section_height"])
        concrete_height = float(self.geometry["concrete_height"])

        kerf_depth = float(self.geometry["kerf_depth"])
        kerf_length = float(self.geometry["kerf_length"])
        kerf_forewood_length = float(self.geometry["kerf_forewood_length"])

        cross_section_top_y = cross_section_height / 2.0
        timber_concrete_interface_y = cross_section_top_y - concrete_height

        kerf_top_y = timber_concrete_interface_y
        kerf_bottom_y = timber_concrete_interface_y - kerf_depth

        kerf_start_x = kerf_forewood_length
        kerf_end_x = kerf_forewood_length + kerf_length

        node_xy_lookup = self.mesh_nodes_df.set_index("node")[["x", "y"]]

        element_regions = [""] * len(self.mesh_elements_df)

        for i in range(len(self.mesh_elements_df)):
            row = self.mesh_elements_df.iloc[i]

            node_ids = [int(row["n_0"]), int(row["n_1"]), int(row["n_2"]), int(row["n_3"])]
            coords = node_xy_lookup.loc[node_ids]

            element_center_x = float(coords["x"].mean())
            element_center_y = float(coords["y"].mean())

            if element_center_y > timber_concrete_interface_y:
                element_regions[i] = "CONCRETE"
            elif (kerf_start_x <= element_center_x <= kerf_end_x) and (kerf_bottom_y <= element_center_y <= kerf_top_y):
                element_regions[i] = "KERF"
            else:
                element_regions[i] = "TIMBER"

        print("DEBUG len(mesh_elements_df) =", len(self.mesh_elements_df))
        print("DEBUG len(element_regions)  =", len(element_regions))

        self.mesh_elements_df["region"] = element_regions

        print("Mesh regions classified.")
        print("  TIMBER elements =", int((self.mesh_elements_df["region"] == "TIMBER").sum()))
        print("  CONCRETE elements =", int((self.mesh_elements_df["region"] == "CONCRETE").sum()))
        print("  KERF elements =", int((self.mesh_elements_df["region"] == "KERF").sum()))


        
    def create_kratos_nodes(self):
        # Create Kratos nodes from the mesh_nodes_df table
        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        if self.mesh_nodes_df is None:
            raise RuntimeError("Mesh error: mesh_nodes_df is not created. Call build_mesh_tables() first.")

        for _, row in self.mesh_nodes_df.iterrows():
            self.structure_mp.CreateNewNode(
            int(row["node"]),
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        )

        print("Kratos nodes created:")
        print("  NumberOfNodes =", self.structure_mp.NumberOfNodes())


    def add_kratos_dofs(self):
        # Add degrees of freedom to all nodes in the main ModelPart
        if self.structure_mp is None:
             raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        KratosMultiphysics.VariableUtils().AddDof(
            KratosMultiphysics.DISPLACEMENT_X,
            KratosMultiphysics.REACTION_X,
            self.structure_mp,
        )
        KratosMultiphysics.VariableUtils().AddDof(
            KratosMultiphysics.DISPLACEMENT_Y,
            KratosMultiphysics.REACTION_Y,
            self.structure_mp,
        )
        KratosMultiphysics.VariableUtils().AddDof(
            KratosMultiphysics.DISPLACEMENT_Z,
            KratosMultiphysics.REACTION_Z,
            self.structure_mp,
        )

        print("Kratos DOFs added.")


    def assign_kratos_material_properties(self):
        # Assign material properties and constitutive laws to Kratos Properties
        if self.structure_mp is None:
         raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        self.validate_materials()

        if self.timber_law is None or self.concrete_law is None or self.kerf_law is None:
            raise ValueError("Constitutive law error: timber_law, concrete_law and kerf_law must be set.")

        if self.timber_law not in constitutive_laws_tcc:
            raise KeyError(f"Constitutive law error: timber_law '{self.timber_law}' not found. Available: {list(constitutive_laws_tcc.keys())}")
        if self.concrete_law not in constitutive_laws_tcc:
            raise KeyError(f"Constitutive law error: concrete_law '{self.concrete_law}' not found. Available: {list(constitutive_laws_tcc.keys())}")
        if self.kerf_law not in constitutive_laws_tcc:
            raise KeyError(f"Constitutive law error: kerf_law '{self.kerf_law}' not found. Available: {list(constitutive_laws_tcc.keys())}")

        timber_data = wood[self.wood]
        concrete_data = concrete[self.concrete]

        props_timber = self.structure_mp.GetProperties()[1]
        props_concrete = self.structure_mp.GetProperties()[2]
        props_kerf = self.structure_mp.GetProperties()[3]

        # Timber elastic properties
        props_timber.SetValue(KratosMultiphysics.DENSITY, float(timber_data["density"]))
        props_timber.SetValue(KratosMultiphysics.YOUNG_MODULUS, float(timber_data["youngs_modulus"]))
        props_timber.SetValue(KratosMultiphysics.SHEAR_MODULUS, float(timber_data["shear_modulus"]))
        props_timber.SetValue(KratosMultiphysics.POISSON_RATIO, float(timber_data["poissons_ratio"]))
        props_timber.SetValue(KratosMultiphysics.THICKNESS, float(timber_data["thickness"]))

        # Concrete elastic properties
        props_concrete.SetValue(KratosMultiphysics.DENSITY, float(concrete_data["density"]))
        props_concrete.SetValue(KratosMultiphysics.YOUNG_MODULUS, float(concrete_data["youngs_modulus"]))
        props_concrete.SetValue(KratosMultiphysics.SHEAR_MODULUS, float(concrete_data["shear_modulus"]))
        props_concrete.SetValue(KratosMultiphysics.POISSON_RATIO, float(concrete_data["poissons_ratio"]))
        props_concrete.SetValue(KratosMultiphysics.THICKNESS, float(concrete_data["thickness"]))

        # Kerf properties. For now: same as concrete material values
        props_kerf.SetValue(KratosMultiphysics.DENSITY, float(concrete_data["density"]))
        props_kerf.SetValue(KratosMultiphysics.YOUNG_MODULUS, float(concrete_data["youngs_modulus"]))
        props_kerf.SetValue(KratosMultiphysics.SHEAR_MODULUS, float(concrete_data["shear_modulus"]))
        props_kerf.SetValue(KratosMultiphysics.POISSON_RATIO, float(concrete_data["poissons_ratio"]))
        props_kerf.SetValue(KratosMultiphysics.THICKNESS, float(concrete_data["thickness"]))

        # Constitutive laws from registry
        props_timber.SetValue(KratosMultiphysics.CONSTITUTIVE_LAW, constitutive_laws_tcc[self.timber_law]())
        props_concrete.SetValue(KratosMultiphysics.CONSTITUTIVE_LAW, constitutive_laws_tcc[self.concrete_law]())
        props_kerf.SetValue(KratosMultiphysics.CONSTITUTIVE_LAW, constitutive_laws_tcc[self.kerf_law]())

        print("Kratos material properties assigned:")
        print("  Timber law =", props_timber.GetValue(KratosMultiphysics.CONSTITUTIVE_LAW))
        print("  Concrete law =", props_concrete.GetValue(KratosMultiphysics.CONSTITUTIVE_LAW))
        print("  Kerf law =", props_kerf.GetValue(KratosMultiphysics.CONSTITUTIVE_LAW))
    
    def assign_timber_plasticity_parameters(self):
        # Assign plasticity parameters for timber if enabled
        if not self.use_plasticity_parameters:
            print("Plasticity parameters skipped (use_plasticity_parameters = False).")
            return

        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        self.validate_materials()

        timber_data = wood[self.wood]
        props_timber = self.structure_mp.GetProperties()[1]

        required = [
            "yield_stress",
            "isotropic_hardening_modulus",
            "saturation_yield_stress",
            "hardening_exponent",
        ]

        for k in required:
            if k not in timber_data:
                raise KeyError(f"Plasticity error: missing key '{k}' in wood['{self.wood}']")

        props_timber.SetValue(KratosMultiphysics.KratosGlobals.GetVariable("YIELD_STRESS"), float(timber_data["yield_stress"]))
        props_timber.SetValue(KratosMultiphysics.KratosGlobals.GetVariable("ISOTROPIC_HARDENING_MODULUS"), float(timber_data["isotropic_hardening_modulus"]))
        props_timber.SetValue(KratosMultiphysics.KratosGlobals.GetVariable("EXPONENTIAL_SATURATION_YIELD_STRESS"), float(timber_data["saturation_yield_stress"]))
        props_timber.SetValue(KratosMultiphysics.KratosGlobals.GetVariable("HARDENING_EXPONENT"), float(timber_data["hardening_exponent"]))

        print("Timber plasticity parameters assigned.")


    def create_kratos_elements_by_region(self):
        #Create Kratos elements using region based Properties.
        #Properties[1] = TIMBER, Properties[2] = CONCRETE, Properties[3] = KERF.
        #Also assigns elements and nodes to SubModelParts 'TIMBER', 'CONCRETE', 'KERF'.
    
        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        if self.mesh_elements_df is None:
            raise RuntimeError("Mesh error: mesh_elements_df is not created. Call build_mesh_tables() first.")

        if "region" not in self.mesh_elements_df.columns:
            raise RuntimeError("Mesh error: region is not classified. Call classify_mesh_elements_regions() first.")

        if not self.structure_mp.HasSubModelPart("TIMBER"):
            self.structure_mp.CreateSubModelPart("TIMBER")
        if not self.structure_mp.HasSubModelPart("CONCRETE"):
            self.structure_mp.CreateSubModelPart("CONCRETE")
        if not self.structure_mp.HasSubModelPart("KERF"):
            self.structure_mp.CreateSubModelPart("KERF")

        mp_timber = self.structure_mp.GetSubModelPart("TIMBER")
        mp_concrete = self.structure_mp.GetSubModelPart("CONCRETE")
        mp_kerf = self.structure_mp.GetSubModelPart("KERF")

        timber_node_ids = set()
        concrete_node_ids = set()
        kerf_node_ids = set()

        for _, element_row in self.mesh_elements_df.iterrows():
            element_id = int(element_row["element"])
            node_ids = [
                int(element_row["n_0"]),
                int(element_row["n_1"]),
                int(element_row["n_2"]),
                int(element_row["n_3"]),
            ]
            region = str(element_row["region"])

            if region == "TIMBER":
                element_properties = self.structure_mp.GetProperties()[1]
                region_mp = mp_timber
                timber_node_ids.update(node_ids)

            elif region == "CONCRETE":
                element_properties = self.structure_mp.GetProperties()[2]
                region_mp = mp_concrete
                concrete_node_ids.update(node_ids)

            elif region == "KERF":
                element_properties = self.structure_mp.GetProperties()[3]
                region_mp = mp_kerf
                kerf_node_ids.update(node_ids)

            else:
                element_properties = self.structure_mp.GetProperties()[1]
                region_mp = mp_timber
                timber_node_ids.update(node_ids)

            created_element = self.structure_mp.CreateNewElement(
                self.element_name,
                element_id,
                node_ids,
                element_properties,
            )
            region_mp.AddElement(created_element)

        if timber_node_ids:
            mp_timber.AddNodes(list(timber_node_ids))
        if concrete_node_ids:
            mp_concrete.AddNodes(list(concrete_node_ids))
        if kerf_node_ids:
            mp_kerf.AddNodes(list(kerf_node_ids))

        print("Kratos elements created by region.")
        print("  Total elements =", self.structure_mp.NumberOfElements())
        print("  TIMBER elements =", mp_timber.NumberOfElements())
        print("  CONCRETE elements =", mp_concrete.NumberOfElements())
        print("  KERF elements =", mp_kerf.NumberOfElements())

    def debug_region_mapping_in_kratos(self):
        #Debug check: verify that elements really use Properties[1]/[2]/[3] and that region SubModelParts contain the expected Property IDs.
        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created.")

        from collections import Counter

        # 1) Count which Property IDs are used by all elements in the main ModelPart
        element_property_ids = [element.Properties.Id for element in self.structure_mp.Elements]
        property_id_counts = Counter(element_property_ids)
        print("PropertyId distribution (all elements) =", dict(property_id_counts))

        # 2) For each region SubModelPart, verify the Property IDs used by its elements
        expected_property_id_by_region = {"TIMBER": 1, "CONCRETE": 2, "KERF": 3}

        for region_name, expected_property_id in expected_property_id_by_region.items():
            if not self.structure_mp.HasSubModelPart(region_name):
                print(region_name, "SubModelPart missing")
                continue

            region_model_part = self.structure_mp.GetSubModelPart(region_name)

            region_element_property_ids = {element.Properties.Id for element in region_model_part.Elements}

            print(
                region_name,
                "elements =", region_model_part.NumberOfElements(),
                "prop_ids =", region_element_property_ids,
                "expected =", expected_property_id,
            )
    def apply_support_conditions(self):
        #Apply supports using the external function stored in self.support['apply'].
        self.validate_support()

        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")
        if self.support_mp is None:
            raise RuntimeError("Kratos error: support_mp is not created. Call create_submodelparts() first.")
        if self.mesh_nodes_df is None:
            raise RuntimeError("Mesh error: mesh_nodes_df is not created. Call build_mesh_tables() first.")

        apply_support_callback = self.support.get("apply", None)
        if apply_support_callback is None:
            print("Support not applied. No 'apply' function defined in support dictionary.")
            return
        apply_support_callback(self)
        print("Support applied via external function.")

    def apply_load_conditions(self, step):
        # Apply loads using external function stored in self.loadcase["apply"].
        self.validate_loadcase()

        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")
        if self.load_mp is None:
            raise RuntimeError("Kratos error: load_mp is not created. Call create_submodelparts() first.")
        if self.mesh_nodes_df is None:
            raise RuntimeError("Mesh error: mesh_nodes_df is not created. Call build_mesh_tables() first.")

        loadcase_apply_callback = self.loadcase.get("apply", None)
        if loadcase_apply_callback is None:
            print("Load not applied. No 'apply' function defined in loadcase dictionary.")
            return

        loadcase_apply_callback(self, step)


    def build_solver_parameters(self):
        # Build Kratos Parameters from self.solver_settings dict
        self.validate_solver_settings()

        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        solver_settings_dict = self.solver_settings
        model_part_name = self.structure_mp.Name

        simulation_parameters = KratosMultiphysics.Parameters(f"""
        {{
            "problem_data" : {{
            "problem_name"  : "tcc_generated",
            "parallel_type" : "OpenMP",
            "echo_level"    : {int(solver_settings_dict["echo_level"])},
            "start_time"    : {float(solver_settings_dict["start_time"])},
            "end_time"      : {float(solver_settings_dict["end_time"])}
            }},
            "solver_settings" : {{
                "time_stepping" : {{
                    "time_step" : {float(solver_settings_dict["time_step"])}
                }},
                "solver_type"     : "{solver_settings_dict["solver_type"]}",
                "analysis_type"   : "{solver_settings_dict["analysis_type"]}",
                "model_part_name" : "{model_part_name}",
                "domain_size"     : {int(solver_settings_dict["domain_size"])},
                "echo_level"      : {int(solver_settings_dict["echo_level"])},
                "model_import_settings" : {{
                    "input_type"     : "use_input_model_part",
                    "input_filename" : "tcc_generated"
                }},
                "convergence_criterion"       : "{solver_settings_dict["convergence_criterion"]}",
                "residual_relative_tolerance" : {float(solver_settings_dict["residual_relative_tolerance"])},
                "residual_absolute_tolerance" : {float(solver_settings_dict["residual_absolute_tolerance"])},
                "max_iteration"               : {int(solver_settings_dict["max_iteration"])},
                "rotation_dofs"               : false,
                "volumetric_strain_dofs"      : false
            }}
        }}
        """)
        return simulation_parameters
    
    def solve_current_step(self, current_step):
        # Run Kratos analysis once for the current load state
        if self.kratos_model is None:
            raise RuntimeError("Kratos error: kratos_model is not created. Call create_kratos_model() first.")

        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        simulation_parameters = self.build_solver_parameters()

        self.structure_mp.ProcessInfo[KratosMultiphysics.STEP] = int(current_step)

        analysis = StructuralMechanicsAnalysis(self.kratos_model, simulation_parameters)
        analysis.Run()

        print("Solve finished.")
        print("  step =", int(current_step))

    def create_vtk_output(self, output_folder="results/vtk"):
        # Create a VTK output process that writes nodal and gauss point results.
        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")
        
        vtk_settings = KratosMultiphysics.Parameters(r'''
    {
        "model_part_name": "model_part_structure",
        "output_control_type": "step",
        "output_interval": 1,
        "output_path": "REPLACE_FOLDER",
        "file_format": "binary",

        "nodal_solution_step_data_variables": [
            "DISPLACEMENT",
            "REACTION"
        ],

        "gauss_point_variables_in_elements": [
            "CAUCHY_STRESS_VECTOR"
        ]
    }
    ''')

        vtk_settings["output_path"].SetString(str(output_folder))

        self.vtk_output = VtkOutputProcess(self.kratos_model, vtk_settings)

        self.vtk_output.ExecuteInitialize()
        print("VTK output created.")
        print("  folder =", output_folder)

    def write_vtk_output(self):
        # Write VTK for the current solution step.
        if not hasattr(self, "vtk_output") or self.vtk_output is None:
            raise RuntimeError("VTK error: vtk_output is not created. Call create_vtk_output() first.")

        self.vtk_output.PrintOutput()
        print("VTK output written for current step.")

    def create_vtk_outputs_by_region(self, output_base_folder="results/vtk", extra_nodal_variables=None, extra_gauss_point_variables=None):
        """
        Create 3 VTK output processes (TIMBER, CONCRETE, KERF) that write into one case folder.
        Files are prefixed with: timber_, concrete_, kerf_.
        """

        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        nodal_solution_step_variables = ["DISPLACEMENT", "REACTION"]
        gauss_point_variables_in_elements = ["CAUCHY_STRESS_VECTOR"]

        if extra_nodal_variables:
            nodal_solution_step_variables += list(extra_nodal_variables)

        if extra_gauss_point_variables:
            gauss_point_variables_in_elements += list(extra_gauss_point_variables)

        base_case_name = "tcc_element"
        case_index = 1
        case_folder = os.path.join(output_base_folder, f"{base_case_name}_{case_index}")
        while os.path.exists(case_folder):
            case_index += 1
            case_folder = os.path.join(output_base_folder, f"{base_case_name}_{case_index}")

        os.makedirs(case_folder, exist_ok=True)

        self.vtk_output_by_region = {}

        for region_name in ["TIMBER", "CONCRETE", "KERF"]:
            vtk_parameters = KratosMultiphysics.Parameters("{}")

            vtk_parameters.AddString("model_part_name", f"{self.structure_mp.Name}.{region_name}")
            vtk_parameters.AddString("output_control_type", "step")
            vtk_parameters.AddInt("output_interval", 1)
            vtk_parameters.AddString("output_path", case_folder)
            vtk_parameters.AddString("file_format", "binary")
            vtk_parameters.AddString("custom_name_prefix", f"{region_name.lower()}_")
            vtk_parameters.AddBool("write_deformed_configuration", True)

            vtk_parameters.AddEmptyArray("nodal_solution_step_data_variables")
            for variable_name in nodal_solution_step_variables:
                vtk_parameters["nodal_solution_step_data_variables"].Append(
                    KratosMultiphysics.Parameters(f"\"{variable_name}\"")
                )

            vtk_parameters.AddEmptyArray("gauss_point_variables_in_elements")
            for variable_name in gauss_point_variables_in_elements:
                vtk_parameters["gauss_point_variables_in_elements"].Append(
                    KratosMultiphysics.Parameters(f"\"{variable_name}\"")
                )

            vtk_process = VtkOutputProcess(self.kratos_model, vtk_parameters)
            vtk_process.ExecuteInitialize()
            self.vtk_output_by_region[region_name] = vtk_process

        print("VTK outputs created in folder:", case_folder)



    def write_vtk_outputs_by_region(self):
        # Write VTK output for all region processes.

        if not hasattr(self, "vtk_output_by_region") or self.vtk_output_by_region is None:
            raise RuntimeError("VTK error: vtk_output_by_region is not created. Call create_vtk_outputs_by_region() first.")

        for region_name, vtk_process in self.vtk_output_by_region.items():
            vtk_process.PrintOutput()

        print("VTK outputs written for TIMBER, CONCRETE, KERF.")
        
    
"""

#input
tcc_element_1 = TimberConcreteCompositeElement()
tcc_element_1.wood = "birk"
tcc_element_1.concrete = "C25/30"
tcc_element_1.geometry = geometry_tcc["default"]
tcc_element_1.support = support_tcc.get("single_span_beam")  # returns None if typo
tcc_element_1.loadcase = loadcase_tcc.get("lc4_uniform_line_load")
tcc_element_1.solver_settings = solver_tcc.get("default_static_nonlinear")
tcc_element_1.timber_law = "linear_elastic_plane_strain_2d"
tcc_element_1.concrete_law = "linear_elastic_plane_strain_2d"
tcc_element_1.kerf_law = "linear_elastic_plane_strain_2d"




#call methods
tcc_element_1.validate_materials()
tcc_element_1.validate_geometry()
print("Materials and geometry validated successfully.")
tcc_element_1.call_geometry_properties()
tcc_element_1.validate_support()
print("Support validated successfully.")
tcc_element_1.validate_loadcase()
print("Loadcase validated successfully.")
tcc_element_1.validate_solver_settings()
print("Solver settings validated successfully.")
tcc_element_1.create_kratos_model()
tcc_element_1.add_kratos_variables()
tcc_element_1.create_submodelparts()
tcc_element_1.build_mesh_tables()
tcc_element_1.classify_mesh_elements_regions()
tcc_element_1.create_kratos_nodes()
tcc_element_1.add_kratos_dofs()
tcc_element_1.assign_kratos_material_properties()
tcc_element_1.assign_timber_plasticity_parameters()
tcc_element_1.create_kratos_elements_by_region()
tcc_element_1.debug_region_mapping_in_kratos()
tcc_element_1.apply_support_conditions()
tcc_element_1.apply_load_conditions(step=1)
#tcc_element_1.create_vtk_output(output_folder="results/vtk")

tcc_element_1.create_vtk_outputs_by_region(output_base_folder="results/vtk")
tcc_element_1.solve_current_step(current_step=1)
tcc_element_1.write_vtk_outputs_by_region()



"""