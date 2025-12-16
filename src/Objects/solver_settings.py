# src/Objects/solver_settings.py

# Solver settings presets for the TCC element.

solver_tcc = {
    "default_static_nonlinear": {
        "solver_type": "Static",
        "analysis_type": "non_linear",
        "domain_size": 2,
        "echo_level": 0,

        "time_step": 1.0,
        "start_time": 0.0,
        "end_time": 1.0,

        "convergence_criterion": "residual_criterion",
        "residual_relative_tolerance": 1e-4,
        "residual_absolute_tolerance": 1e-9,

        "max_iteration": 30
    }
}
