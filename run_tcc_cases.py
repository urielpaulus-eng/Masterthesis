from src.Class.TimberConcreteCompositeElement import TimberConcreteCompositeElement

from src.Objects.geometry import geometry_tcc
from src.Objects.supports import support_tcc
from src.Objects.loadcases import loadcase_tcc
from src.Objects.solver_settings import solver_tcc


extra_gp = ["EQUIVALENT_PLASTIC_STRAIN", "PLASTIC_STRAIN_VECTOR"]


def run_element_inline(tcc_element: TimberConcreteCompositeElement, step: int):
    tcc_element.validate_materials()
    tcc_element.validate_geometry()
    tcc_element.call_geometry_properties()

    tcc_element.validate_support()
    tcc_element.validate_loadcase()
    tcc_element.validate_solver_settings()

    tcc_element.create_kratos_model()
    tcc_element.add_kratos_variables()
    tcc_element.create_submodelparts()

    tcc_element.build_mesh_tables()
    tcc_element.classify_mesh_elements_regions()

    tcc_element.create_kratos_nodes()
    tcc_element.add_kratos_dofs()

    tcc_element.assign_kratos_material_properties()
    tcc_element.assign_timber_plasticity_parameters()

    tcc_element.create_kratos_elements_by_region()
    tcc_element.apply_support_conditions()
    tcc_element.apply_load_conditions(step=step)

    tcc_element.create_vtk_outputs_by_region(
        output_base_folder="results/vtk",
        extra_gauss_point_variables=extra_gp
    )

    tcc_element.solve_current_step(current_step=step)
    tcc_element.write_vtk_outputs_by_region()


tcc_element_1 = TimberConcreteCompositeElement()
tcc_element_1.case_name = "tcc_element_1"
tcc_element_1.wood = "oak"
tcc_element_1.concrete = "C30/37"
tcc_element_1.geometry = geometry_tcc["default"]
tcc_element_1.support = support_tcc["single_span_beam"]
tcc_element_1.loadcase = loadcase_tcc["lc4_uniform_line_load"]
tcc_element_1.solver_settings = solver_tcc["default_static_nonlinear"]
tcc_element_1.timber_law = "linear_elastic_plane_strain_2d"
tcc_element_1.concrete_law = "linear_elastic_plane_strain_2d"
tcc_element_1.kerf_law = "linear_elastic_plane_strain_2d"

run_element_inline(tcc_element_1, step=1)


tcc_element_2 = TimberConcreteCompositeElement()
tcc_element_2.case_name = "tcc_element_2"
tcc_element_2.wood = "birk"
tcc_element_2.concrete = "C25/30"
tcc_element_2.geometry = geometry_tcc["default"]
tcc_element_2.support = support_tcc["single_span_beam"]
tcc_element_2.loadcase = loadcase_tcc["lc4_uniform_line_load"]
tcc_element_2.solver_settings = solver_tcc["default_static_nonlinear"]
tcc_element_2.timber_law = "linear_elastic_plane_strain_2d"
tcc_element_2.concrete_law = "linear_elastic_plane_strain_2d"
tcc_element_2.kerf_law = "linear_elastic_plane_strain_2d"

run_element_inline(tcc_element_2, step=1)
