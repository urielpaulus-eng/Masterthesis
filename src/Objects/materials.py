# Esto es un diccionario de todos los materiales
# Units:
# density: kg/m3
# moduli: Pa
# thickness: mm

wood = {

    "oak": {
        "density": 10000.0,
        "youngs_modulus": 14400.0,
        "shear_modulus": 900.0,
        "poissons_ratio": 0.3,
        "thickness": 100.0,


        # neue Einträge für J2-Plastizität:
        "yield_stress": 15000.0,   # Streckgrenze in [Pa] (anpassen!)
        "isotropic_hardening_modulus": 1000,   # isotrope Verfestigung [Pa]
        "saturation_yield_stress":   100,   # Sättigungs-Streckgrenze [Pa]
        "hardening_exponent": 10     # Exponent für exp. Verfestigung
    },

    "birk": {
        "density": 10000.0,
        "youngs_modulus": 14400.0,
        "shear_modulus": 900.0,
        "poissons_ratio": 0.3,
        "thickness": 100.0,


        "yield_stress": 15000.0,   # Streckgrenze in [Pa] (anpassen!)
        "isotropic_hardening_modulus": 1000,   # isotrope Verfestigung [Pa]
        "saturation_yield_stress":   100,   # Sättigungs-Streckgrenze [Pa]
        "hardening_exponent": 10     # Exponent für exp. Verfestigung
    },

    "beech": {
       "density": 10000.0,
        "youngs_modulus": 14400.0,
        "shear_modulus": 900.0,
        "poissons_ratio": 0.3,
        "thickness": 100.0,


        "yield_stress": 15000.0,   # Streckgrenze in [Pa] (anpassen!)
        "isotropic_hardening_modulus": 1000,   # isotrope Verfestigung [Pa]
        "saturation_yield_stress":   100,   # Sättigungs-Streckgrenze [Pa]
        "hardening_exponent": 10     # Exponent für exp. Verfestigung
    }
    
}


concrete = {

    "C25/30": {
        "density": 400,
        "youngs_modulus": 30000.0,      # beispielhaft
        "shear_modulus": 12500.0,      # beispielhaft
        "poissons_ratio": 0.2,
        "thickness": 100.0,
    },

    "C30/37": { #Werte noch einfügen
        "density": 400,
        "youngs_modulus": 30000.0,      # beispielhaft
        "shear_modulus": 12500.0,      # beispielhaft
        "poissons_ratio": 0.2,
        "thickness": 100.0,

    }

}

#print(wood["oak"]["density"])
