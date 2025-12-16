# src/Objects/supports.py
# TimberConcreteComposite = tcc
# Support definitions for the TCC element.
# Values are kept simple and will be interpreted later in the solver setup.


import KratosMultiphysics

def apply_support_single_span_beam(tcc):
    #Fixed left bottom, moveable right bottom.
    min_x = float(tcc.mesh_nodes_df["x"].min())
    max_x = float(tcc.mesh_nodes_df["x"].max())
    min_y = float(tcc.mesh_nodes_df["y"].min())

    left_bottom_row = tcc.mesh_nodes_df[(tcc.mesh_nodes_df["x"] == min_x) & (tcc.mesh_nodes_df["y"] == min_y)].iloc[0]
    right_bottom_row = tcc.mesh_nodes_df[(tcc.mesh_nodes_df["x"] == max_x) & (tcc.mesh_nodes_df["y"] == min_y)].iloc[0]

    fixed_node_id = int(left_bottom_row["node"])
    moveable_node_id = int(right_bottom_row["node"])

    fixed_mp = tcc.support_mp.CreateSubModelPart("fixed_support")
    moveable_mp = tcc.support_mp.CreateSubModelPart("moveable_support")

    fixed_mp.AddNodes([fixed_node_id])
    moveable_mp.AddNodes([moveable_node_id])

    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_X, True, fixed_mp.Nodes)
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Y, True, fixed_mp.Nodes)
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Z, True, fixed_mp.Nodes)

    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Y, True, moveable_mp.Nodes)
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Z, True, moveable_mp.Nodes)

    print("Supports applied.")
    print("  support_type =", "single_span_beam")
    print("  fixed_support_node_id =", fixed_node_id)
    print("  moveable_support_node_id =", moveable_node_id)


def apply_support_beam_compression_tension(tcc):
    """Placeholder."""
    print("Support not applied. support_type 'beam_compression_tension' not implemented yet.")




support_tcc = {
    "single_span_beam": {
        "type": "single_span_beam",
        "apply": apply_support_single_span_beam,
    },
    "beam_compression_tension": {
        "type": "beam_compression_tension",
        "apply": apply_support_beam_compression_tension,
    }
}