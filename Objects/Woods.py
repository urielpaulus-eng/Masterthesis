# Esto es un diccionario de todos los materiales

wood = {

    "oak": {
        "density": 10000.0,
        "E": 14400.0,
        "G": 900.0,
        "nu": 0.3,
        "thickness": 100.0,  # mm


        # neue Einträge für J2-Plastizität:
        "yield_stress":       15000.0,   # Streckgrenze in [Pa] (anpassen!)
        "H_iso":              1000,   # isotrope Verfestigung [Pa]
        "yield_stress_sat":   100,   # Sättigungs-Streckgrenze [Pa]
        "hardening_exponent": 10     # Exponent für exp. Verfestigung
    },

    "birk": {
        "density": 10000.0,
        "E": 14400.0,
        "G": 900.0,
        "nu": 0.3,
        "thickness": 100.0,  # mm


        # neue Einträge für J2-Plastizität:
        "yield_stress":       15000.0,   # Streckgrenze in [Pa] (anpassen!)
        "H_iso":              1000,   # isotrope Verfestigung [Pa]
        "yield_stress_sat":   100,   # Sättigungs-Streckgrenze [Pa]
        "hardening_exponent": 10     # Exponent für exp. Verfestigung
    },

    "beech": {
        "density": 15000.0,
        "E": 14400.0,
        "G": 900.0,
        "nu": 0.3,
        "thickness": 100.0,  # mm


        # neue Einträge für J2-Plastizität:
        "yield_stress":       15000.0,   # Streckgrenze in [Pa] (anpassen!)
        "H_iso":              1000,   # isotrope Verfestigung [Pa]
        "yield_stress_sat":   100,   # Sättigungs-Streckgrenze [Pa]
        "hardening_exponent": 10     # Exponent für exp. Verfestigung
    }

}

#print(wood["oak"]["density"])
