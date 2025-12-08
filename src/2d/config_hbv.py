"""
config_hbv.py

Zentrale Parameterdatei für dein Glulam / HBV Modell.
Hier kannst du später für die Parameterstudie alles ändern,
ohne im eigentlichen FE-Skript zu suchen.
"""

# Geometrie in mm und Elementanzahlen
GEOMETRY = {
    # Basisgeometrie
    "l": 3000.0,     # Balkenlänge in x-Richtung [mm]
    "b": 300.0,      # Gesamtbauhöhe in y-Richtung (Holz + Beton) [mm]
    "h": 0.0,        # Dicke in z-Richtung (2D-Modell → 0) [mm]

    # Aufteilung der Gesamtbauhöhe b
    "b_timber": 220.0,   # Holzdicke (unterer Teil) [mm]
    "b_concrete": 80.0,  # Betondicke (oberer Teil) [mm]

      # Diskretisierung
    "n_el_x": 98,
    "n_el_y": 14,
    "n_el_z": 0,

    # Kervengeometrie
    "kerf_depth": 60.0,         # Kerventiefe d_n [mm]
    "kerf_length": 400.0,       # Kervenlänge l_n [mm]
    "lv": 500.0,                # Vorholzlänge l_v [mm]
    "kerf_angle_deg": 90.0,     # Kervenflanken-Winkel α_n [°]
}

assert abs(GEOMETRY["b_timber"] + GEOMETRY["b_concrete"] - GEOMETRY["b"]) < 1e-6, \
    "b_timber + b_concrete stimmt nicht mit Gesamt-b überein!"


# Materialparameter Holz (aktuell wie im Betreuer-Code)
MATERIAL_TIMBER = {
    "density": 0.0,
    "E": 14400.0,
    "G": 900.0,
    "nu": 0.3,
    "thickness": 100.0,  # mm

    # neue Einträge für J2-Plastizität:
    "yield_stress":       3.0e6,   # Streckgrenze in [Pa] (anpassen!)
    "H_iso":              1.0e8,   # isotrope Verfestigung [Pa]
    "yield_stress_sat":   4.0e6,   # Sättigungs-Streckgrenze [Pa]
    "hardening_exponent": 10.0     # Exponent für exp. Verfestigung
}

# Materialparameter Beton (Platzhalterwerte, kannst du später anpassen)
MATERIAL_CONCRETE = {
    "density": 0.0,
    "E": 30000.0,      # beispielhaft
    "G": 12500.0,      # beispielhaft
    "nu": 0.2,
    "thickness": 100.0,
}

# Materialparameter für Kerf-Bereich
# vorerst gleiche Werte wie Beton, später anpassbar
MATERIAL_KERVE= {
    "density": 0.0,
    "E": 30000.0,
    "G": 12500.0,
    "nu": 0.2,
    "thickness": 100.0,
}

# Auflager-Typ für das Modell
SUPPORT = {
    # "single_span_beam" oder "beam_compression_tension"
    "type": "single_span_beam",
}

# Lastfall-Einstellungen
LOADCASE = {
    # "lc1" = 4-Punkt-Biegung
    # "lc2" = Zug/Druck
    # "lc3" = Einzellast
    # NEU: "lc4" = gleichmäßige Linienlast (z.B. Eigengewicht)
    "case": "lc4",

    # hier interpretieren wir max_value als Linienlast q_y [N/mm]
    # Beispiel: -0.03 N/mm entspricht -30 N/m (wenn mm als Einheit verwendet) Faktor 1000
    "application": "force_controlled",  # für lc4 wird application nicht benutzt
    "max_value": -900.0,                 # negativ = nach unten  900N/mm entspricht 900kN/m

    "n_steps": 1,  # am Anfang gerne 1 lassen
}

