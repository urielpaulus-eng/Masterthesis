# src/Objects/loadcases.py
# TimberConcreteComposite = tcc
# Loadcase definitions for the TCC element.
# Values are kept simple and will be interpreted later in the solver setup.

loadcase_tcc = {
    "lc1_4point_bending": {
        "type": "lc1",
        "application": "force_controlled",
        "max_value": 0.0,
        "n_steps": 1
    },
    "lc2_compression_tension": {
        "type": "lc2",
        "application": "deformation_controlled",
        "max_value": 0.0,
        "n_steps": 1
    },
    "lc3_single_point_load": {
        "type": "lc3",
        "application": "force_controlled",
        "max_value": 0.0,
        "n_steps": 1
    },
    "lc4_uniform_line_load": {
        "type": "lc4",
        "application": "force_controlled",
        "max_value": -800.0,
        "n_steps": 2
    }
}