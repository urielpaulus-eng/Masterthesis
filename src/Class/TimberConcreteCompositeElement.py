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

    def create_kratos_elements(self):
        # Create Kratos elements from the mesh_elements_df table
        if self.structure_mp is None:
            raise RuntimeError("Kratos error: structure_mp is not created. Call create_kratos_model() first.")

        if self.mesh_elements_df is None:
            raise RuntimeError("Mesh error: mesh_elements_df is not created. Call build_mesh_tables() first.")

        # Temporary: use Properties[1] for all elements
        props = self.structure_mp.GetProperties()[1]

        for _, row in self.mesh_elements_df.iterrows():
            element_id = int(row["element"])
            node_ids = [
                int(row["n_0"]),
                int(row["n_1"]),
                int(row["n_2"]),
                int(row["n_3"]),
            ]

            self.structure_mp.CreateNewElement(
                "SmallDisplacementElement2D4N",
                element_id,
                node_ids,
                props,
            )

        print("Kratos elements created:")
        print("  NumberOfElements =", self.structure_mp.NumberOfElements())

    from src.Objects.constitutive_laws import constitutive_laws_tcc

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




#input
tcc_element_1 = TimberConcreteCompositeElement()
tcc_element_1.wood = "oak"
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
tcc_element_1.create_kratos_nodes()
tcc_element_1.add_kratos_dofs()
tcc_element_1.create_kratos_elements()
tcc_element_1.assign_kratos_material_properties()

