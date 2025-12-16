# src/Objects/geometry.py
# TimberConcreteComposite = tcc
# Geometry definitions for the TCC element. As Dictionary.
# Units in mm.

geometry_tcc = {
    "default": {

        # Beam / specimen size
        "tcc_element_length": 3000.0,
        "cross_section_height": 300.0,  #cross_section_height = total height → timber_height + concrete_height
        "cross_section_width": 0.0, # 2D-Model → 0

        # Layer heights
        "timber_height": 220.0,
        "concrete_height": 80.0,


        # Kerf / notch geometry
        "kerf_depth": 60.0,
        "kerf_length": 400.0,
        "kerf_forewood_length": 500.0,
        "kerf_flank_angle_deg": 90.0,    # unit in °

        # Mesh resolution → Number of elements in each direction for the calculation.
        "num_mesh_elements_x": 72,
        "num_mesh_elements_y": 10,
        "num_mesh_elements_z": 0,
    }
}