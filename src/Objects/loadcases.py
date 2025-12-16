# src/Objects/loadcases.py
# TimberConcreteComposite = tcc
# Loadcase definitions for the TCC element.
# Values are kept simple and will be interpreted later in the solver setup.


import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA

def apply_load_lc4_uniform_line_load(tcc, step):
    # Uniform line load q_y along top edge. Converted to equivalent point loads at top nodes.

    max_value = float(tcc.loadcase["max_value"])
    n_steps = int(tcc.loadcase["n_steps"])

    load_factor = float(step) / float(n_steps)
    q_y = max_value * load_factor  # User decides sign. Negative means downward.

    y_top = float(tcc.mesh_nodes_df["y"].max())
    tol = 1.0e-9 * max(1.0, abs(y_top))

    top_nodes_df = tcc.mesh_nodes_df[(tcc.mesh_nodes_df["y"] - y_top).abs() <= tol].copy()
    if len(top_nodes_df) < 2:
        print("Load lc4 not applied. Less than 2 top nodes found.")
        return

    top_nodes_df = top_nodes_df.sort_values("x")
    top_node_ids = [int(nid) for nid in top_nodes_df["node"].tolist()]
    top_x = [float(x) for x in top_nodes_df["x"].tolist()]

    if not tcc.load_mp.HasSubModelPart("lc4_uniform_line_load"):
        lc4_mp = tcc.load_mp.CreateSubModelPart("lc4_uniform_line_load")
    else:
        lc4_mp = tcc.load_mp.GetSubModelPart("lc4_uniform_line_load")

    lc4_mp.AddNodes(top_node_ids)

    trib_lengths = []
    for i in range(len(top_x)):
        if i == 0:
            trib_lengths.append(0.5 * (top_x[i + 1] - top_x[i]))
        elif i == len(top_x) - 1:
            trib_lengths.append(0.5 * (top_x[i] - top_x[i - 1]))
        else:
            trib_lengths.append(0.5 * (top_x[i + 1] - top_x[i - 1]))

    try:
        load_properties = tcc.structure_mp.GetProperties()[2]
    except Exception:
        load_properties = tcc.structure_mp.GetProperties()[1]

    created_or_updated = 0

    for node_id, L_i in zip(top_node_ids, trib_lengths):
        Fy = q_y * float(L_i)

        condition_id = 4000000000 + int(node_id)

        if tcc.structure_mp.HasCondition(condition_id):
            condition = tcc.structure_mp.GetCondition(condition_id)
        else:
            condition = lc4_mp.CreateNewCondition(
                "PointLoadCondition2D1N",
                condition_id,
                [node_id],
                load_properties,
            )

        condition.SetValue(SMA.POINT_LOAD, [0.0, float(Fy), 0.0])
        created_or_updated += 1

    print("Load applied.")
    print("  loadcase_type =", "lc4_uniform_line_load")
    print("  step =", int(step))
    print("  q_y =", float(q_y))
    print("  top_nodes =", len(top_node_ids))
    print("  conditions_updated =", created_or_updated)

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
        "type": "lc4_uniform_line_load",
        "application": "force_controlled",
        "max_value": -11524.0,
        "n_steps": 10,
        "apply": apply_load_lc4_uniform_line_load,
    }
}