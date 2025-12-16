# src/Objects/constitutive_laws.py
# Constitutive Law definition for the diferent layers

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics import ConstitutiveLawsApplication as CLA

# Factory functions. Each call returns a new law object.

constitutive_laws_tcc = {
    "linear_elastic_plane_strain_2d": lambda: SMA.LinearElasticPlaneStrain2DLaw(),
    "j2_plasticity_plane_strain_2d": lambda: CLA.SmallStrainJ2PlasticityPlaneStrain2DLaw(),
}
