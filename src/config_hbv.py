"""
config_hbv.py

Zentrale Parameterdatei für dein Glulam / HBV Modell.
Hier kannst du später für die Parameterstudie alles ändern,
ohne im eigentlichen FE-Skript zu suchen.
"""

# Geometrie in mm und Elementanzahlen
GEOMETRY = {
    "length": 5000.0,    #mm
    "width": 600.0,      # mm
    "height": 0.0,       # mm (2D-Modell, also 0)
    "n_el_x": 72,        # Elemente über die Länge
    "n_el_y": 10,        # Elemente über die Breite
    "n_el_z": 0,         # Elemente über die Höhe (0 = 2D)

     # --- Kerve-Geometrie (2D) ---
    "kerven_tiefe_dn": 60.0,       # Kerventiefe d_n [mm] (nach unten in Holz)
    "kerven_laenge_ln": 200.0,     # Kervenlänge l_n [mm] (in x-Richtung)
    "vorholz_laenge_lv": 500.0,    # Vorholzlänge l_v [mm] (Abstand von linkem Auflager)
    "kerven_winkel_deg": 90.0,     # Kervenflanken-Winkel α_n [°], erstmal 90°
}

# Materialparameter Holz (aktuell wie im Betreuer-Code)
MATERIAL_TIMBER = {
    "density": 0.0,
    "E": 14400.0,
    "G": 900.0,
    "nu": 0.3,
    "thickness": 100.0,  # mm
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
    # "lc1" (4-Punkt-Biegung), "lc2" (Zug/Druck), "lc3" (Einzellast)
    "case": "lc1",

    # "deformation_controlled" (Weg vorgeben) oder "force_controlled" (Kraft vorgeben)
    "application": "deformation_controlled",

    # Maximalwert (mm oder N, je nach application)
    "max_value": 140.0,

    # Anzahl der Schritte (1 = so wie jetzt)
    "n_steps": 1,
}
