from __future__ import print_function, absolute_import, division    # sorgt für Kompatibilität mit Python 2.6/2.7 (altes Kratos-Umfeld)

'''Timber Engineering - KratosMultiphysics'''  # Beschreibung des Projekts (String, hat keine Funktion)

# Help for Data_Structure: Kratos/docs/pages/Kratos/For_Users/Crash_course/Data_Structure
# Good example: Kratos/applications/StructuralMechanicsApplication/tests/test_prebuckling_analysis.py
# Obige Kommentare: Hinweise auf Dokumentation / Beispiele in Kratos

'''Import packages'''
# KratosMultiphysics
import KratosMultiphysics  # Hauptmodul von Kratos
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis  # allgemeine Strukturanalyse-Klasse (hier nicht direkt benutzt)
from KratosMultiphysics.vtk_output_process import VtkOutputProcess  # Prozess zum Schreiben von VTK-Dateien
from KratosMultiphysics.StructuralMechanicsApplication.displacement_control_with_direction_process import AssignDisplacementControlProcess  # Prozess für wegkontrollierte Lasten

from KratosMultiphysics import ConstitutiveLawsApplication  # Modul für erweiterte Materialgesetze

# Stuff
import pandas as pd  # Pandas für Tabellen (DataFrames)
import numpy as np   # NumPy für numerische Operationen

'''Import functions'''
from def_node_element_creator import define_nodes  # eigene Funktion: erzeugt DataFrames mit Knoten
from def_node_element_creator import define_quadrilateralN4  # eigene Funktion: erzeugt DataFrames mit 4-Knoten-Elementen
from def_plot import plot_data_2D  # eigene Funktion: 2D-Plot (z.B. Spannungsverteilung)

'''Glulam Beam with Solid 2D element'''

'Structure'
# First part of the code consist of def functions.
# Second part of the code consist the application of functions 
# Kommentar: erster Teil = Definition der Funktionen, zweiter Teil = Ausführen/Verwenden der Funktionen

'to do'
# Check if load can be apllied to node without element
# Check solver settings and their influence
# Constitutive Law for timber (2D-asymetric stiffness matrix)
# Plastification of Timber on Compressionside
# Model Tension failure
# Monte-Carlo Simulation
# rename nodes to nodes
#
#
# Kommentar: Liste offener Aufgaben / Ideen für spätere Erweiterungen

'Input'
l = 10800               # Länge des Balkens in mm
b = 600                 # Breite (b) des Querschnitts in mm
h = 0                   # Höhe (h) in mm -> 0, weil 2D-Modell (Plane-Strain)
n_element_x = 72         # Anzahl der Elemente in Längsrichtung (muss gerade sein laut Kommentar)
n_element_y = 10         # Anzahl der Elemente über die Breite
n_element_z = 0         # Anzahl der Elemente in z-Richtung (0, weil 2D)

'Dataframe with nodes and elements'
df_nodes, df_nodes_xyz, df_nodes_number = define_nodes(l,b,h,n_element_x,n_element_y,n_element_z)  # erzeugt Knoten-DataFrames basierend auf Geometrie und Diskretisierung
df_elements, df_elements_number = define_quadrilateralN4(n_element_x,n_element_y,n_element_z)      # erzeugt Element-DataFrames für 4-Knoten-2D-Elemente

####################################################################################################
'First part: Define function for setup the numerical model - Pre-processing'
####################################################################################################

'Model and Modelparts'
def model_modelpart():  # Funktion zum Erzeugen von Model und ModelParts
    # Create Model
    model = KratosMultiphysics.Model()  # globales Kratos-Model-Objekt (Container für alle ModelParts)
    # Create Modelpart "model_part_structure"
    model_part_structure = model.CreateModelPart("model_part_structure")  # Haupt-ModelPart für die Struktur
    model_part_structure.SetBufferSize(2)  # setzt Anzahl der Zeit-Schritte im Speicher (hier 2, auch für statisch nötig)
    # Create SubModelPart "boundary_condition_support_model_part_structure"
    boundary_condition_support_model_part_structure = model_part_structure.CreateSubModelPart("boundary_condition_support_model_part_structure")  # SubModelPart für Lager
    # Create SubModelPart "boundary_condition_load_model_part_structure"
    boundary_condition_load_model_part_structure = model_part_structure.CreateSubModelPart("boundary_condition_load_model_part_structure")  # Ober-SubModelPart für Lasten
    # Create SubModelPart "boundary_condition_load_lc1_part_structure"
    boundary_condition_load_lc1_model_part_structure = boundary_condition_load_model_part_structure.CreateSubModelPart("boundary_condition_load_lc1_model_part_structure")  # SubModelPart für Lastfall 1
    # Create SubModelPart "boundary_condition_load_lc2_part_structure"
    boundary_condition_load_lc2_model_part_structure = boundary_condition_load_model_part_structure.CreateSubModelPart("boundary_condition_load_lc2_model_part_structure")  # SubModelPart für Lastfall 2
    # Create SubModelPart "boundary_condition_load_lc3_part_structure"
    boundary_condition_load_lc3_model_part_structure = boundary_condition_load_model_part_structure.CreateSubModelPart("boundary_condition_load_lc3_model_part_structure")  # SubModelPart für Lastfall 3
    return model, model_part_structure, boundary_condition_support_model_part_structure, boundary_condition_load_model_part_structure, boundary_condition_load_lc1_model_part_structure, boundary_condition_load_lc2_model_part_structure, boundary_condition_load_lc3_model_part_structure  # gibt alle relevanten Model/ModelPart-Objekte zurück

'Variables'
def variables(model_part_structure):  # Funktion zum Registrieren der benötigten Variablen im ModelPart
    # Define variables to ModelPart
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)  # Verschiebungsvektor (ux, uy, uz) an Knoten aktivieren
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.REACTION)      # Reaktionskräfte an Knoten aktivieren
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.LOAD_FACTOR)  # Skalierungsfaktor für Lasten (für Displacement-Control)
    model_part_structure.AddNodalSolutionStepVariable(KratosMultiphysics.StructuralMechanicsApplication.PRESCRIBED_DISPLACEMENT)  # Vorgeschriebene Verschiebung an Knoten

'Material and Constitutive Law'
def material(model_part_structure):  # Funktion zur Definition der Materialeigenschaften
    # Define Material to ModelPart
    model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.DENSITY,0)  # Dichte = 0 (Eigengewicht wird ignoriert)
    model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.YOUNG_MODULUS,14400)        # Elastizitätsmodul E (N/mm²)
    model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.SHEAR_MODULUS,900)          # Schubmodul G (N/mm²)
    model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.POISSON_RATIO,0.3)          # Querdehnzahl ν
    model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.THICKNESS,100.0)            # Dicke des 2D-Elements in z-Richtung (mm)

    constitutive_law = KratosMultiphysics.StructuralMechanicsApplication.LinearElasticPlaneStrain2DLaw()  # linear-elastisches 2D-Ebenen-Verzerrungs-Gesetz (Plane Strain)

    # Plastification Variant 1 (not working)
    # Die auskommentierten Zeilen sind ein Versuch, ein nichtlineares/plastisches Material zu verwenden
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.DENSITY,0)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.YOUNG_MODULUS,14400)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.SHEAR_MODULUS,900)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.POISSON_RATIO,0.3)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.THICKNESS,100.0)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.SOFTENING_TYPE,0)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.HARDENING_CURVE,3)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.YIELD_STRESS_TENSION,40.0)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.YIELD_STRESS_COMPRESSION,30)
    # model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.FRACTURE_ENERGY ,10000000000.0)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.FRACTURE_ENERGY_COMPRESSION ,1000000000)
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.STRESS_DAMAGE_CURVE ,[-40])
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.STRAIN_DAMAGE_CURVE ,[5])
    # model_part_structure.GetProperties()[1].SetValue(ConstitutiveLawsApplication.DAMAGE_TENSION ,0.5)
    
    # constitutive_law = KratosMultiphysics.KratosGlobals.GetConstitutiveLaw("SmallStrainDplusDminusDamageRankineRankine2D").Clone()
    # constitutive_law = KratosMultiphysics.KratosGlobals.GetConstitutiveLaw("SmallStrainDplusDminusDamageVonMisesVonMises2D").Clone()

    # Define Constitutive Law to ModelPart
    model_part_structure.GetProperties()[1].SetValue(KratosMultiphysics.CONSTITUTIVE_LAW,constitutive_law)  # weist den Properties[1] das Materialgesetz zu

'Geometry (Nodes and Elements) and DOFs' 'Degrees of Freedom'
def geometry_DOF(model_part_structure,df_nodes,df_elements):  # Funktion zur Geometrie-Erzeugung und DOF-Zuweisung
    # Create Nodes in ModelPart
    for i in range(df_nodes.shape[0]):   # Schleife über alle Knoten im DataFrame
        model_part_structure.CreateNewNode(int(df_nodes["node"][i]),df_nodes["x"][i],df_nodes["y"][i],df_nodes["z"][i])  # erzeugt jeden Knoten mit ID und Koordinaten
    # Output in Terminal of Nodes in ModelPart
    # for node in model_part_structure.Nodes:
    #     print(node.Id, node.X, node.Y, node.Z)
    # Create Elements in ModelPart
    for i in range(df_elements.shape[0]):   # Schleife über alle Elemente im DataFrame
        model_part_structure.CreateNewElement("SmallDisplacementElement2D4N",int(df_elements["element"][i]),[df_elements["n_0"][i],df_elements["n_1"][i],df_elements["n_2"][i],df_elements["n_3"][i]],model_part_structure.GetProperties()[1])  # erzeugt jedes 4-Knoten-2D-Element mit Knoten-IDs und Materialeigenschaften
    # Output in Terminal of Nodes in ModelPart
    # for element in model_part_structure.Elements:
    #     print(element.Id,element.Properties.Id,element.GetNode(0).Id,element.GetNode(1).Id,element.GetNode(2).Id,element.GetNode(3).Id)
    # Define DOFs to ModelPart
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.DISPLACEMENT_X, KratosMultiphysics.REACTION_X, model_part_structure)  # fügt ux-Freiheitsgrad + zugehörige Reaktion an allen Knoten hinzu
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.DISPLACEMENT_Y, KratosMultiphysics.REACTION_Y, model_part_structure)  # fügt uy-Freiheitsgrad + Reaktion hinzu
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.DISPLACEMENT_Z, KratosMultiphysics.REACTION_Z, model_part_structure)  # fügt uz-Freiheitsgrad + Reaktion hinzu
    KratosMultiphysics.VariableUtils().AddDof(KratosMultiphysics.StructuralMechanicsApplication.LOAD_FACTOR, KratosMultiphysics.StructuralMechanicsApplication.PRESCRIBED_DISPLACEMENT, model_part_structure)  # fügt Load-Factor-DOF + prescribed displacement hinzu

'Boundary Condition - Support'
# Define boundary conditions for supports - single-span beam
def boundary_condition_support_single_span_beam(boundary_condition_support_model_part_structure):  # Funktion für Lagerbedingungen eines Einfeldträgers
    # Create SubModelPart "boundary_condition_fixed_support_model_part_structure"
    boundary_condition_fixed_support_model_part_structure = boundary_condition_support_model_part_structure.CreateSubModelPart("boundary_condition_fixed_support_model_part_structure")  # SubModelPart für das feste Lager
    # Create SubModelPart "boundary_condition_moveable_support_model_part_structure"
    boundary_condition_moveable_support_model_part_structure = boundary_condition_support_model_part_structure.CreateSubModelPart("boundary_condition_moveable_support_model_part_structure")  # SubModelPart für das verschiebliche Lager
    # Define nodes for supports of single-span beam
    boundary_condition_fixed_support_model_part_structure.AddNodes([1500250350])  # fügt den linken Lagerknoten (ID aus deinem Nummerierungssystem) in das feste Lager-SubModelPart hinzu
    boundary_condition_moveable_support_model_part_structure.AddNodes([1500250350+n_element_x*1000000])  # fügt rechten Lagerknoten (am anderen Ende) in das verschiebliche Lager-SubModelPart hinzu
    # Function for applying boundary condtions (fixed support) to nodes of a submodelpart
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_X, True, boundary_condition_fixed_support_model_part_structure.Nodes)  # sperrt Verschiebung in x-Richtung am festen Lager
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Y, True, boundary_condition_fixed_support_model_part_structure.Nodes)  # sperrt Verschiebung in y-Richtung am festen Lager
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Z, True, boundary_condition_fixed_support_model_part_structure.Nodes)  # sperrt Verschiebung in z-Richtung (für 2D meist irrelevant)
    # Function for applying boundary condtions (moveable support) to nodes of a submodelpart
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Y, True, boundary_condition_moveable_support_model_part_structure.Nodes)  # sperrt uy am verschieblichen Lager
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Z, True, boundary_condition_moveable_support_model_part_structure.Nodes)  # sperrt uz am verschieblichen Lager (2D)
    # ux bleibt am verschieblichen Lager frei -> horizontale Verschiebung möglich (Loslager)

# Define boundary conditions for supports - beam under compression/tension
def boundary_condition_support_beam_compression_tension(boundary_condition_support_model_part_structure):  # Funktion für Lagerung bei reinem Zug/Druckstab
    # Create SubModelPart "boundary_condition_fixed_support_model_part_structure"
    boundary_condition_fixed_support_model_part_structure = boundary_condition_support_model_part_structure.CreateSubModelPart("boundary_condition_fixed_support_model_part_structure")  # SubModelPart für fixierten Querschnitt
    # Define nodes for support of beam under compression/tension
    for i in range(int(n_element_y/2+1)):  # Schleife über halbe Querschnittshöhe + Mitte
        if i == 0:  # erster Durchlauf: Mittelknoten
            boundary_condition_fixed_support_model_part_structure.AddNodes([1500250350])  # mittlerer Knoten in Querschnittsmitte
        else:
            boundary_condition_fixed_support_model_part_structure.AddNodes([int(1500250350+i*1000)])   # Knoten auf der einen Seite der Mitte → bedeutet die komplette Seite ist an jedem Knoten gehalten.
            boundary_condition_fixed_support_model_part_structure.AddNodes([int(1500250350-i*1000)])   # Knoten auf der anderen Seite der Mitte
    # Function for applying boundary condtions (fixed support) to nodes of a submodelpart
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_X, True, boundary_condition_fixed_support_model_part_structure.Nodes)  # sperrt ux für alle Lagerknoten
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Y, True, boundary_condition_fixed_support_model_part_structure.Nodes)  # sperrt uy für alle Lagerknoten
    KratosMultiphysics.VariableUtils().ApplyFixity(KratosMultiphysics.DISPLACEMENT_Z, True, boundary_condition_fixed_support_model_part_structure.Nodes)  # sperrt uz für alle Lagerknoten
#Ist dass dann nicht eingespannt?

'Boundary Condition - Load'
# Define boundary conditions for load - loadcase 1 (lc1), 4-point-bending
def boundary_condition_load_lc1(model_part_structure,boundary_condition_load_lc1_model_part_structure,step,load_X,load_Y,load_Z):  # Funktion zur Definition von Lastfall 1 (Vierpunkt-Biegung)
    # Define nodes for loadcase 1 (lc1)
    boundary_condition_load_lc1_model_part_structure.AddNodes([int(1500250350 + n_element_x/3*1000000 - n_element_y/2*1000)])      # erster Lastknoten (bei 1/3 der Balkenlänge und auf einer Seite der Höhe)
    boundary_condition_load_lc1_model_part_structure.AddNodes([int(1500250350 + n_element_x*2/3*1000000 - n_element_y/2*1000)])    # zweiter Lastknoten (bei 2/3 der Balkenlänge)
    # Force controlled - Apply point load to nodes
    # Die folgenden Zeilen wären für eine kraftgesteuerte Last (auskommentiert)
    # for node in boundary_condition_load_lc1_model_part_structure.Nodes:
    #     condition_load_model_part_structure = boundary_condition_load_lc1_model_part_structure.CreateNewCondition("PointLoadCondition2D1N",step*10000000000+node.Id,[node.Id],model_part_structure.GetProperties()[1])
    #     condition_load_model_part_structure.SetValue(KratosMultiphysics.StructuralMechanicsApplication.POINT_LOAD, [load_X,load_Y,load_Z])
    # Displacement controlled - Apply point load to nodes
    for node in boundary_condition_load_lc1_model_part_structure.Nodes:  # Schleife über die beiden Lastknoten
        condition_load_model_part_structure = boundary_condition_load_lc1_model_part_structure.CreateNewCondition("DisplacementControlCondition3D1N",step*10000000000+node.Id,[node.Id],model_part_structure.GetProperties()[1])  # erzeugt eine Condition für wegkontrollierte Last an diesem Knoten (Condition-ID basiert auf step und Knoten-ID)
        condition_load_model_part_structure = AssignDisplacementControlProcess(model,  # erstellt einen Prozess, der die Verschiebung steuert (nutzt globale Variable 'model')
                                            KratosMultiphysics.Parameters("""{
                                                    "model_part_name": "model_part_structure.boundary_condition_load_model_part_structure.boundary_condition_load_lc1_model_part_structure",
                                                    "direction"       : "y",
                                                    "point_load_value": 1,
                                                    "prescribed_displacement_value" : "140"}
                                                """))  # Parameter: SubModelPart-Name, Verschiebungsrichtung (y), skalierende Last, Zielverschiebung (140 mm)
        condition_load_model_part_structure.ExecuteInitializeSolutionStep()  # initialisiert den Displacement-Control-Prozess für diesen Lastschritt

# Define boundary conditions for load - loadcase 2 (lc2), compression/tension load
def boundary_condition_load_lc2(model_part_structure,boundary_condition_load_lc2_model_part_structure,step,load_X,load_Y,load_Z):  # Funktion für Lastfall 2 (reiner Druck/Zug über die Querschnittsfläche)
    # Define nodes for loadcase 2 (lc2)
    for i in range(int(n_element_y/2+1)):  # Schleife über halbe Höhe
        if i == 0:  # Mitte der Höhe
            boundary_condition_load_lc2_model_part_structure.AddNodes([int(1500250350+n_element_x*1000000)])  # Knoten in der Mitte der rechten Stirnseite
        else:
            boundary_condition_load_lc2_model_part_structure.AddNodes([int(1500250350+n_element_x*1000000+i*1000)])   # Knoten oberhalb der Mitte
            boundary_condition_load_lc2_model_part_structure.AddNodes([int(1500250350+n_element_x*1000000-i*1000)])   # Knoten unterhalb der Mitte
    # Apply point load to nodes
    for node in boundary_condition_load_lc2_model_part_structure.Nodes:  # Schleife über alle Lastknoten
        condition_load_model_part_structure = boundary_condition_load_lc2_model_part_structure.CreateNewCondition("PointLoadCondition2D1N",step*10000000000+node.Id,[node.Id],model_part_structure.GetProperties()[1])  # erstellt eine Punklast-Condition für jeden Knoten
        condition_load_model_part_structure.SetValue(KratosMultiphysics.StructuralMechanicsApplication.POINT_LOAD, [load_X,load_Y,load_Z])  # setzt die Lastkomponenten (X, Y, Z) an der Condition

# Define boundary conditions for load - loadcase 3 (lc3), load at single node
def boundary_condition_load_lc3(model_part_structure,boundary_condition_load_lc3_model_part_structure,step,load_X,load_Y,load_Z):  # Funktion für Lastfall 3 (Punktlast an einem einzelnen Knoten)
    # Define nodes for loadcase 3 (lc3)
    boundary_condition_load_lc3_model_part_structure.AddNodes([int(1522250350)])  # fügt den speziellen Lastknoten hinzu
    # Apply point load to node
    for node in boundary_condition_load_lc3_model_part_structure.Nodes:  # Schleife (hier nur 1 Knoten)
        condition_load_model_part_structure = boundary_condition_load_lc3_model_part_structure.CreateNewCondition("PointLoadCondition2D1N",step*10000000000+node.Id,[node.Id],model_part_structure.GetProperties()[1])  # erstellt Punktlast-Condition
        condition_load_model_part_structure.SetValue(KratosMultiphysics.StructuralMechanicsApplication.POINT_LOAD, [load_X,load_Y,load_Z])  # setzt Lastvektor an diesem Knoten


#BIS HIER WURDE NACHVOLLZOGEN 19.11.2025

'Solver stategy - under construction'
# Not clear what the single function actual do and how they work -> Testing
# Compare with .py in Kratos/applications/StructuralMechanicsApplication/tests
# Kommentar: Hinweis, dass Solverstrategie noch experimentell ist

# Define and build solver
def apply_solver(model_part_structure):  # Funktion zum Aufbauen und Ausführen des Solvers
    linear_solver = KratosMultiphysics.SkylineLUFactorizationSolver()   # wählt einen Skyline-LU-Löser für das Gleichungssystem
    builder_and_solver = KratosMultiphysics.ResidualBasedBlockBuilderAndSolver(linear_solver)   # erstellt Builder+Solver, der die Systemmatrix aufbaut und mit dem LU-Löser löst
    scheme = KratosMultiphysics.StructuralMechanicsApplication.StructuralMechanicsStaticScheme(KratosMultiphysics.Parameters("{}"))     # statisches Lösungsschema (wie die Gleichungen in jedem Schritt aktualisiert werden)
    # scheme = KratosMultiphysics.ResidualBasedIncrementalUpdateStaticScheme()
    convergence_criterion = KratosMultiphysics.ResidualCriteria(1e-14,1e-20)  # Konvergenzkriterium basierend auf Residuen (sehr kleine Toleranzen)
       
    # Define linear strategy to solve the problem
    max_iters = 20                    # maximale Anzahl von Newton-Raphson-Iterationen
    compute_reactions = True          # Reaktionskräfte sollen berechnet werden
    reform_step_dofs = True           # DOFs werden nach jedem Schritt neu aufgebaut (nützlich bei sich ändernden Randbedingungen)
    calculate_norm_dx = False         # Norm der Inkremente nicht berechnen (hier ausgeschaltet)
    move_mesh_flag = True             # Knotenkoordinaten werden mit Verschiebungen aktualisiert (verschobene Geometrie)
    # strategy = KratosMultiphysics.ResidualBasedLinearStrategy(
    #                                                             model_part_structure,
    #                                                             scheme,
    #                                                             builder_and_solver,
    #                                                             compute_reactions,
    #                                                             reform_step_dofs,
    #                                                             calculate_norm_dx,
    #                                                             move_mesh_flag)
    strategy = KratosMultiphysics.ResidualBasedNewtonRaphsonStrategy(  # definiert eine Newton-Raphson-Strategie (nichtlinearer statischer Solver)
                                                                model_part_structure,
                                                                scheme,
                                                                convergence_criterion,
                                                                builder_and_solver,
                                                                max_iters,
                                                                compute_reactions,
                                                                reform_step_dofs,
                                                                move_mesh_flag)
    convergence_criterion.SetEchoLevel(0)  # unterdrückt Ausgaben des Konvergenzkriteriums (0 = still)
    strategy.SetEchoLevel(0)              # unterdrückt Ausgaben der Strategie (0 = still)
    strategy.Initialize()                 # initialisiert die Strategie (z.B. DOF-Listen, Matrizenstrukturen)
    strategy.Check()                      # führt interne Konsistenzprüfungen durch
    strategy.Solve()                      # startet den eigentlichen Lösungsprozess (Assemblierung, Iteration, Ergebnisberechnung)

'Output - General'
# General Output in Terminal
# print(model_part_structure)
# Kommentar: wäre eine Text-Ausgabe der Struktur, ist aber auskommentiert

'Output - VTK'
def apply_output_vtk(model):  # Funktion zum Schreiben der Ergebnisse in VTK-Dateien
    vtk_output_process = VtkOutputProcess(model,  # erstellt einen VTK-Ausgabeprozess für das gegebene Model
                                            KratosMultiphysics.Parameters("""{
                                                    "model_part_name"                    : "model_part_structure",
                                                    "output_control_type"                : "step",
                                                    "output_interval"                    : 1,
                                                    "file_format"                        : "ascii",
                                                    "output_precision"                   : 7,
                                                    "output_sub_model_parts"             : false,
                                                    "write_deformed_configuration"       : true,
                                                    "output_path"                        : "vtk_output_glulam_solid_2D",
                                                    "save_output_files_in_folder"        : false,
                                                    "nodal_solution_step_data_variables" : ["DISPLACEMENT","REACTION"],
                                                    "gauss_point_variables_extrapolated_to_nodes": ["PK2_STRESS_VECTOR"],
                                                    "gauss_point_variables_in_elements": ["PK2_STRESS_VECTOR"]
                                                    
                                                }
                                                """)
                                            )  # Parameter-JSON: steuert, welche Größen und wie sie geschrieben werden
    vtk_output_process.ExecuteInitialize()             # initialisiert den Output-Prozess
    vtk_output_process.ExecuteBeforeSolutionLoop()     # wird vor der (theoretischen) Lösungsschleife aufgerufen
    vtk_output_process.ExecuteInitializeSolutionStep() # initialisiert Ausgabe für aktuellen Schritt
    vtk_output_process.PrintOutput()                   # schreibt die Ergebnisdateien (VTK)
    vtk_output_process.ExecuteFinalizeSolutionStep()   # Abschlussarbeiten nach der Ausgabe eines Schrittes
    vtk_output_process.ExecuteFinalize()               # schließt den Prozess ab (z.B. Dateien)

        
'Output system response quantity (SRQ)'
def apply_output_SRQ(model_part_structure):  # Funktion zum Sammeln von Systemantworten (z.B. Verschiebungen, Spannungen) in DataFrames
    # Deformation in nodes
    list_node_deformation = []  # leere Liste für Knotendeformationen
    node: KratosMultiphysics.Node
    for node in model_part_structure.Nodes:        # Schleife über alle Knoten
        # Deformation in historical container
        dict_node_deformation = {}                 # Dictionary für einen Knoten
        dict_node_deformation.update({'node':node.Id, 'stress_x': node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y)})  # speichert Knoten-ID und uy-Verschiebung (Name 'stress_x' ist irreführend)
        list_node_deformation.append(dict_node_deformation)  # hängt Dictionary an Liste an
    df_node_deformation = pd.DataFrame(list_node_deformation)  # konvertiert die Liste in ein Pandas-DataFrame
    # Stess in gauss points
    extrapolation_parameters = KratosMultiphysics.Parameters("""  # definiert Parameter für Spannungs-Extrapolation von Integrationspunkten zu Knoten
    {
    "model_part_name"            : "model_part_structure",
    "echo_level"                 : 2,
    "average_variable"           : "NODAL_AREA",
    "area_average"               : true,
    "list_of_variables"          : ["PK2_STRESS_VECTOR"],
    "extrapolate_non_historical" : true
    }
    """)
    # Extrapolation stress to nodes
    integration_values_extrapolation_to_nodes_process = KratosMultiphysics.IntegrationValuesExtrapolationToNodesProcess(model_part_structure,extrapolation_parameters)  # erstellt Prozess für Extrapolation von Integrationspunktwerten zu Knoten
    integration_values_extrapolation_to_nodes_process.Execute()  # führt Extrapolation aus
    list_node_PK2_stress = []  # leere Liste für Knoten-Spannungen
    node: KratosMultiphysics.Node
    for node in model_part_structure.Nodes:        # Schleife über alle Knoten
        # Stresses in non-historical container
        dict_node_PK2_stress = {}                 # Dictionary für einen Knoten
        dict_node_PK2_stress.update({'node':node.Id, 'stress_x': node[KratosMultiphysics.PK2_STRESS_VECTOR][0], 'stress_y': node[KratosMultiphysics.PK2_STRESS_VECTOR][1], 'stress_z': node[KratosMultiphysics.PK2_STRESS_VECTOR][2]})  # liest PK2-Spannungsvektor an Knoten aus (σx, σy, τxy) und speichert ihn
        list_node_PK2_stress.append(dict_node_PK2_stress)  # hängt Dictionary an Liste an
        # Print stresses in non-historical container
        # print(f"node {node.Id}: {node.Has(KratosMultiphysics.PK2_STRESS_VECTOR)}: {node[KratosMultiphysics.PK2_STRESS_VECTOR]}")
    df_node_PK2_stress = pd.DataFrame(list_node_PK2_stress)  # konvertiert Spannungs-Liste in DataFrame
    return df_node_deformation, df_node_PK2_stress           # gibt Verformungs- und Spannungs-DataFrame zurück


####################################################################################################
'Second part: Apply numerical model and start calculation'
####################################################################################################

'Calculate for one step'
model, model_part_structure, boundary_condition_support_model_part_structure, boundary_condition_load_model_part_structure, boundary_condition_load_lc1_model_part_structure, boundary_condition_load_lc2_model_part_structure, boundary_condition_load_lc3_model_part_structure = model_modelpart()  # erzeugt Model und alle ModelParts/SubModelParts
variables(model_part_structure)                     # registriert notwendige Variablen im Struktursystem
material(model_part_structure)                      # definiert Materialparameter und Materialgesetz
geometry_DOF(model_part_structure,df_nodes,df_elements)  # erzeugt Knoten, Elemente und DOFs im ModelPart
boundary_condition_support_single_span_beam(boundary_condition_support_model_part_structure)  # setzt Lagerbedingungen für Einfeldträger
# boundary_condition_support_beam_compression_tension(boundary_condition_support_model_part_structure)  # alternative Lagerdefinition (auskommentiert)
boundary_condition_load_lc1(model_part_structure,boundary_condition_load_lc1_model_part_structure,0,0,85000,0)  # setzt Lastfall 1: Vierpunkt-Biegung, Schritt 0, Lastkomponenten (X=0, Y=85000, Z=0) – bei Displacement-Control wird Y hier nicht verwendet
# boundary_condition_load_lc2(model_part_structure,boundary_condition_load_lc2_model_part_structure,0,0,-100000/(n_element_y+1),0)  # alternative Last: verteilter Druck/Zug (auskommentiert)
# boundary_condition_load_lc3(model_part_structure,boundary_condition_load_lc3_model_part_structure,0,0,85000,0)  # alternative Last: Einzelpunktlast (auskommentiert)
apply_solver(model_part_structure)                  # führt den Solver aus und berechnet Verschiebungen/Spannungen/Reaktionen
apply_output_vtk(model)                             # schreibt VTK-Ausgabedateien für Visualisierung in ParaView
df_node_deformation, df_node_PK2_stress = apply_output_SRQ(model_part_structure)  # sammelt Verformungen und Spannungen in DataFrames

'Calculate for multi-step'
# # Define steps
# step = 0 # Running step
# end_step = 10 # Number of steps
# # Define load
# load_max = 85
# load_min = load_max/end_step
# load_step = np.linspace(load_min,load_max,end_step)
# # List for output
# list_load_deformation_curve = []
# # Setup Model
# model, model_part_structure, boundary_condition_support_model_part_structure, boundary_condition_load_model_part_structure, boundary_condition_load_lc1_model_part_structure, boundary_condition_load_lc2_model_part_structure, boundary_condition_load_lc3_model_part_structure = model_modelpart()
# variables(model_part_structure)
# material(model_part_structure)
# geometry_DOF(model_part_structure,df_nodes,df_elements)
# boundary_condition_support_single_span_beam(boundary_condition_support_model_part_structure)
# # Start calculation loop
# while step < end_step:
#     # Updating step
#     step += 1
#     model_part_structure.ProcessInfo[KratosMultiphysics.STEP] = step
#     # Boundary conditions        
#     boundary_condition_load_lc1(model_part_structure,boundary_condition_load_lc1_model_part_structure,step,0,load_min*1000,0)
#     # Solver
#     apply_solver(model_part_structure)
#     # Ouput - VTK
#     apply_output_vtk(model)
#     # Output - Load-Deformation-Curve
#     node_id_mid_span = 1500250350+n_element_x/2*1000000
#     node = model_part_structure.GetNode(int(node_id_mid_span))
#     node_deformation  = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y)
#     dict_load_deformation_curve = {}
#     dict_load_deformation_curve.update({'deformation':node_deformation, 'load': load_step[step-1]})
#     list_load_deformation_curve.append(dict_load_deformation_curve)
# Kommentar zu diesem Block:
# - Ist auskommentiert.
# - Würde eine inkrementelle Mehrschritt-Analyse durchführen (step von 1 bis end_step).
# - In jedem Schritt: Last erhöhen, lösen, VTK ausgeben, mittlere Durchbiegung speichern.
# - Am Ende könnte man eine Last-Verformungs-Kurve plotten.

####################################################################################################
'Third part: Post-Processing'
####################################################################################################

'Output - Load-Deformation-Curve'
# df_load_deformation_curve = pd.DataFrame(list_load_deformation_curve)
# plot_data_2D('Load-Deformation-Curve',df_load_deformation_curve['deformation'],df_load_deformation_curve['load'],'Load-Deformation-Curve')
# Kommentar: Dieser Teil würde aus der Liste eine Last-Verformungs-Kurve erstellen und plotten (aktuell auskommentiert)

'Output - stress at midspan'
# Nodes at midspan
node_id_mid_span = 1500250350+n_element_x/2*1000000  # berechnet die Knoten-ID im Feldmittelpunkt des Balkens (in x-Richtung mittig)
nodes_id_mid_span = [node_id_mid_span-1000*n_element_y/2 + i*1000 for i in range(n_element_y+1)]  # erzeugt Liste aller Knoten-IDs entlang einer vertikalen Linie durch den Feldmittelpunkt (über die Höhe)

# Stresses at midspan
df_node_midspan_PK2_stress = df_node_PK2_stress[df_node_PK2_stress['node'].isin(nodes_id_mid_span)].copy()  # filtert Spannungs-DataFrame auf Knoten in der Mittelschnitt-Linie

# Stresses at midspan with node position
df_node_midspan_xyz = df_nodes[df_nodes['node'].isin(nodes_id_mid_span)].copy()  # holt die Koordinaten dieser Knoten aus df_nodes
df_node_midspan_PK2_stress_xyz = pd.merge(df_node_midspan_PK2_stress,df_node_midspan_xyz, on="node", how='left')  # fügt Spannungen und Koordinaten zu einem DataFrame zusammen

# Plot stress distribution
plot_data_2D('Stress Distribution',df_node_midspan_PK2_stress_xyz['stress_x'],df_node_midspan_PK2_stress_xyz['y'],'s_x')  # plottet Spannungsverlauf σx über der y-Koordinate im Feldmittelpunkt

'Output - stress in elements'
# Output elements for evaluation
elements_id_mid_span = [n_element_y/2*(n_element_x-1) + i for i in range(n_element_y)]  # berechnet eine Liste von Element-IDs im Bereich des Feldmittelpunkts (für weitere Auswertungen)
#print(elements_id_mid_span)  # könnte die Element-IDs zur Kontrolle ausgeben
