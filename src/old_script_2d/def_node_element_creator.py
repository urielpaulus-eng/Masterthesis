'''Functions for creating nodes and elements for FEM (mainly Kratos)'''

'''Import packages'''
import numpy as np  # Lade numpy Bibliothek (für Mathematik)
import pandas as pd  # Lade pandas Bibliothek (für Tabellen)
'''Import functions'''


'''Function creating nodes'''
# Definition of coordinate system for a rectangular cross section (4 Nodes):
#   /          /
#  /          /
# o----------o
# |      x   |
# |     /    |
# |    +--y  |
# |    |     | /
# |    z     |/
# o----------o
#
# Definition of node number system (here original coordinate system):
#   x | layer     | y | layer    | z | layer    | 
#   1 | 5 0 0     | 2 |  5 0     | 3 |  5 0     |
#     | - = < 500 |   | - = < 50 |   | - = < 50 |
#     | + = > 500 |   | + = > 50 |   | + = > 50 |
# e.g.:
#   x | layer | y | layer | z | layer | 
#   1 | 5 2 1 | 2 |  4 6  | 3 |  5 8  |
#   -> 1521246358
#
# Function: define_nodes(l,b,h,n_element_x,n_element_y,n_element_z)
# with
# l = length [mm]
# b = width [mm]
# h = heigth [mm]
# n_element_x = number of elements over length
# n_element_y = number of elements over width
# n_element_z = number of elements over heigth



def  define_nodes(l,b,h,n_element_x,n_element_y,n_element_z):  # def = Definiere Funktion mit Namen und Parametern


    """This function creates all nodes in 3D (x, y, z), puts them into pandas DataFrames in different forms:

    df_nodes: node id + coordinates,

    df_nodes_xyz: only coordinates,

    df_nodes_number: only node id.

    """
    # x - layer
    list_nodes_x = []  # [] = Erstelle leere Liste
    n_nodes_x = n_element_x+1  # Rechne Anzahl Knoten: Elemente + 1
    if n_element_x == 0:  # if = Wenn Bedingung erfüllt ist (== bedeutet "ist gleich")
        print("Error - number of elements over the length must be at least n_element_x = 1")  # print() = Zeige Text auf Bildschirm
        exit()  # exit() = Beende Programm sofort
    else:  # else = Sonst (wenn Bedingung NICHT erfüllt)
        l_element_x = l/n_element_x  # / = Division
    dict_nodes_x = {}  # {} = Erstelle leeres Dictionary (Karteikarte)
    dict_nodes_x.update({'layer_x':float(1500), 'x':float(0)})  # .update() = Füge Einträge hinzu; float() = Wandle in Dezimalzahl um
    list_nodes_x.append(dict_nodes_x)  # .append() = Füge Element zur Karteikarte hinzu
    for i in range(int(n_nodes_x-1)):  # for = Schleife (wiederhole); range() = Zahlenfolge; int() = Wandle in Ganzzahl um
        layer_x_1 = 1501 + i  # + = Addition ; layer_x_1 ist die Knotennummerierung
        x_1 = l_element_x + i*l_element_x  # * = Multiplikation ; x_1 ist die Position des Knotens
        dict_nodes_x = {}  # Erstelle neues leeres Dictionary
        dict_nodes_x.update({'layer_x':layer_x_1, 'x':x_1})  # Füge berechnete Werte hinzu
        list_nodes_x.append(dict_nodes_x)  # Füge zur Liste hinzu

    # y - layer
    list_nodes_y = []                      # Erstelle leere Liste für y-Knoten
    n_nodes_y = n_element_y + 1            # Anzahl Knoten in y-Richtung

    if n_element_y == 0:
        b_element_y = 0
    else:
        b_element_y = b / n_element_y      # Elementhöhe (b über n_el_y)

    if n_element_y % 2 == 0:
        # gerade Anzahl Elemente → Knoten genau in der Mitte bei y = 0
        dict_nodes_y = {}
        dict_nodes_y.update({'layer_y': float(250), 'y': float(0)})
        list_nodes_y.append(dict_nodes_y)

        for i in range(int((n_nodes_y - 1) / 2)):
            # oberhalb der Mitte
            layer_y_1 = 251 + i
            y_1 = b_element_y + i * b_element_y
            dict_nodes_y = {}
            dict_nodes_y.update({'layer_y': layer_y_1, 'y': y_1})
            list_nodes_y.append(dict_nodes_y)

            # unterhalb der Mitte
            layer_y_2 = 249 - i
            y_2 = -(b_element_y + i * b_element_y)
            dict_nodes_y = {}
            dict_nodes_y.update({'layer_y': layer_y_2, 'y': y_2})
            list_nodes_y.append(dict_nodes_y)

    elif n_element_y % 2 != 0:
        # ungerade Anzahl Elemente → kein Knoten genau bei y = 0,
        # erster Knoten im Abstand b_element_y/2 von der Mitte
        for i in range(int((n_nodes_y - 1) / 2 + 0.5)):
            # oberhalb der Mitte
            layer_y_1 = 251 + i
            y_1 = b_element_y / 2 + i * b_element_y
            dict_nodes_y = {}
            dict_nodes_y.update({'layer_y': layer_y_1, 'y': y_1})
            list_nodes_y.append(dict_nodes_y)

            # unterhalb der Mitte
            layer_y_2 = 249 - i
            y_2 = -(b_element_y / 2 + i * b_element_y)
            dict_nodes_y = {}
            dict_nodes_y.update({'layer_y': layer_y_2, 'y': y_2})
            list_nodes_y.append(dict_nodes_y)
    

    # z - layer   
    list_nodes_z = []  # Erstelle leere Liste für z-Knoten
    n_nodes_z = n_element_z+1  # Rechne Anzahl Knoten
    if n_element_z == 0:  # Wenn keine Elemente in z-Richtung
        h_element_z = 0  # Elementhöhe = 0
    else:  # Sonst
        h_element_z = h/n_element_z  # Teile Höhe durch Anzahl Elemente
    if n_element_z % 2 == 0:  # Wenn gerade Anzahl
        dict_nodes_z = {}  # Neues Dictionary
        dict_nodes_z.update({'layer_z':float(350), 'z':float(0)})  # Mittelknoten
        list_nodes_z.append(dict_nodes_z)  # Füge hinzu
        for i in range(int((n_nodes_z-1)/2)):  # Schleife für halbe Anzahl
            layer_z_1 = 351 + i  # Layer oben
            z_1 = h_element_z + i*h_element_z  # Position oben
            dict_nodes_z = {}  # Neues Dictionary
            dict_nodes_z.update({'layer_z':layer_z_1, 'z':z_1})  # Speichere Werte
            list_nodes_z.append(dict_nodes_z)  # Füge hinzu
            layer_z_2 = 349 - i  # Layer unten
            z_2 = -(h_element_z + i*h_element_z)  # Negative Position unten
            dict_nodes_z = {}  # Neues Dictionary
            dict_nodes_z.update({'layer_z':layer_z_2, 'z':z_2})  # Speichere Werte
            list_nodes_z.append(dict_nodes_z)  # Füge hinzu
            
    elif n_element_z % 2 != 0:  # Wenn ungerade Anzahl; != bedeutet ungleich; → wenn gerade Zahl ist rest ==0 und wenn ungerade Zahl rest!=0
        for i in range(int((n_nodes_z-1)/2+0.5)):  # Schleife für ungerade Anzahl
            layer_z_1 = 351 + i  # Layer oben
            z_1 = h_element_z/2 + i*h_element_z  # Position oben (Start bei halber Höhe)
            dict_nodes_z = {}  # Neues Dictionary
            dict_nodes_z.update({'layer_z':layer_z_1, 'z':z_1})  # Speichere Werte
            list_nodes_z.append(dict_nodes_z)  # Füge hinzu
            layer_z_2 = 349 - i  # Layer unten
            z_2 = -(h_element_z/2 + i*h_element_z)  # Negative Position unten
            dict_nodes_z = {}  # Neues Dictionary
            dict_nodes_z.update({'layer_z':layer_z_2, 'z':z_2})  # Speichere Werte
            list_nodes_z.append(dict_nodes_z)  # Füge hinzu
            
    df_nodes_x = pd.DataFrame(list_nodes_x)  # pd.DataFrame() = Erstelle Tabelle aus Liste
    df_nodes_y = pd.DataFrame(list_nodes_y)  # Tabelle für y-Knoten
    df_nodes_z = pd.DataFrame(list_nodes_z)  # Tabelle für z-Knoten
    
    # Combination of layers
    df_combined = df_nodes_x.merge(df_nodes_y, how="cross").merge(df_nodes_z, how="cross")  # .merge() = Kombiniere Tabellen; how="cross" = Jede Kombination
    df_combined['node'] = (df_combined['layer_x'].astype(int).astype(str) + df_combined['layer_y'].astype(int).astype(str) + df_combined['layer_z'].astype(int).astype(str)).astype(int)  # ['...'] = Spaltenname; .astype() = Wandle Datentyp um; + = Verknüpfe Text
    df_nodes = df_combined[['node', 'x', 'y', 'z']]  # Wähle nur diese 4 Spalten aus
    df_nodes_xyz = df_nodes.drop(df_nodes.columns[0], axis=1)  # .drop() = Lösche Spalte; .columns[0] = erste Spalte; axis=1 = Spalte (nicht Zeile)
    df_nodes_number = df_nodes.drop([df_nodes.columns[1],df_nodes.columns[2],df_nodes.columns[3]], axis=1)  # Lösche Spalten 1,2,3 (behalte nur node)
    return df_nodes, df_nodes_xyz, df_nodes_number  # return = Gib diese 3 Tabellen als Ergebnis zurück: Eine mit nodes,x,y,z ; andere nur x,y,z; andere nur node

df_nodes, df_nodes_xyz, df_nodes_number = define_nodes(10800,600,0,4,4,0)  # Rufe Funktion auf und speichere die 3 Ergebnisse; 
# df_nodes.to_csv(str("nodes")+".csv", sep='\t', index=False)  # .to_csv() = Speichere als CSV; str() = Wandle in Text; sep='\t' = Trennzeichen Tab; index=False = ohne Zeilennummern
# df_nodes_xyz.to_csv(str("nodes_xyz")+".csv", sep='\t', index=False)
# df_nodes_number.to_csv(str("nodes_number")+".csv", sep='\t', index=False)

'''Function creating element in Kratos'''
# QuadrilateralN4
#       v
#       ^
#       |
# 1-----------2
# |     |     |
# |     |     |
# |     +---- | --> u
# |           |
# |           |
# 0-----------3
#
# Function: define_quadrilateralN4(n_element_x,n_element_y,n_element_z)
# with
# n_element_x = number of elements over length
# n_element_y = number of elements over width (must be 0)
# n_element_z = number of elements over heigth
#
# Element numbering over beam-length:
#   |  ...  |  ...  |  ...  |  ...  |
#   o-------o-------o-------o-------o
#   |   13  |   14  |   15  |   16  |
#   o-------o-------o-------o-------o
#   |   9   |   10  |   11  |   12  |
#   o-------o-------o-------o-------o
#   |   5   |   6   |   7   |   8   |
#   o-------o-------^ x ----o---^v--o
#   |   1   |   2   |   3   |   |->u|
#   o-------o-------o->-----o-------o
#                     y

def define_quadrilateralN4(n_element_x,n_element_y,n_element_z):  # Definiere Funktion für Viereckelemente
    
    if n_element_z > 0:  # > bedeutet "größer als"
        print("Error - this element only works for the xy-layer, for the heigth please use the y-axis (n_element_z = 0)")  # Fehlermeldung
        exit()  # Beende Programm
    
    list_elements = []  # Leere Liste für Elemente
    if n_element_y % 2 == 0:  # Wenn gerade Anzahl Elemente in y-Richtung
        for k in range(n_element_x):  # Schleife über alle Elemente in x-Richtung
            element_1 = n_element_y*(k+1)-n_element_y/2+1  # Berechne Elementnummer → Element 1 ist für Elemente über der Mitte, also y>=250
            if k == 0:  # Wenn erstes Element (k ist 0) also nur beim ersten durchgang der schleife
                n_0_1 = 1500250350  # Knotennummer Ecke 0
                n_1_1 = 1501250350  # Knotennummer Ecke 1
                n_2_1 = 1501251350  # Knotennummer Ecke 2
                n_3_1 = 1500251350  # Knotennummer Ecke 3
            else:  # Für alle anderen Elemente
                n_0_1 = 1500250350+k*1000000  # Addiere k Millionen (für x-Position)
                n_1_1 = 1501250350+k*1000000
                n_2_1 = 1501251350+k*1000000
                n_3_1 = 1500251350+k*1000000               
            dict_elements = {}  # Neues Dictionary für Element
            dict_elements.update({'element':element_1, 'n_0':n_0_1, 'n_1':n_1_1, 'n_2':n_2_1, 'n_3':n_3_1})  # Speichere Elementnummer und 4 Eckknoten
            list_elements.append(dict_elements)  # Füge zur Liste hinzu
            
            element_2 = n_element_y*(k+1)-n_element_y/2  # Zweites Element (unter der Mitte)
            if k == 0:  # Wenn erstes Element
                n_0_2 = 1500249350  # Knotennummern
                n_1_2 = 1501249350
                n_2_2 = 1501250350
                n_3_2 = 1500250350
            else:  # Für andere Elemente
                n_0_2 = 1500249350+k*1000000  # Verschiebe in x-Richtung
                n_1_2 = 1501249350+k*1000000
                n_2_2 = 1501250350+k*1000000
                n_3_2 = 1500250350+k*1000000
            dict_elements = {}  # Neues Dictionary
            dict_elements.update({'element':element_2, 'n_0':n_0_2, 'n_1':n_1_2, 'n_2':n_2_2, 'n_3':n_3_2})  # Speichere Element
            list_elements.append(dict_elements)  # Füge hinzu
            
            for i in range(int(n_element_y/2-1)):  # Schleife für restliche Elemente in y-Richtung
                element_1 = element_1+1  # Erhöhe Elementnummer um 1
                n_0_1 = n_3_1  # Alter Knoten 3 wird neuer Knoten 0
                n_1_1 = n_2_1  # Alter Knoten 2 wird neuer Knoten 1
                n_2_1 = n_1_1+1000  # Neuer Knoten 2 (1000 höher in y-layer)
                n_3_1 = n_0_1+1000  # Neuer Knoten 3
                dict_elements = {}  # Neues Dictionary
                dict_elements.update({'element':element_1, 'n_0':n_0_1, 'n_1':n_1_1, 'n_2':n_2_1, 'n_3':n_3_1})  # Speichere
                list_elements.append(dict_elements)  # Füge hinzu
                element_2 = element_2-1  # Verringere Elementnummer
                n_2_2 = n_1_2  # Verschiebe Knoten
                n_3_2 = n_0_2
                n_0_2 = n_0_2-1000  # 1000 niedriger in y-layer
                n_1_2 = n_1_2-1000

                dict_elements = {}  # Neues Dictionary
                dict_elements.update({'element':element_2, 'n_0':n_0_2, 'n_1':n_1_2, 'n_2':n_2_2, 'n_3':n_3_2})  # Speichere
                list_elements.append(dict_elements)  # Füge hinzu

    df_elements = pd.DataFrame(list_elements)  # Erstelle Tabelle aus Liste
    df_elements = df_elements.sort_values(by='element')  # .sort_values() = Sortiere Tabelle nach Spalte 'element'
    df_elements_number = df_elements.drop([df_elements.columns[1],df_elements.columns[2],df_elements.columns[3],df_elements.columns[4]], axis=1)  # Lösche Spalten 1-4 (behalte nur Elementnummer)
    return df_elements, df_elements_number  # Gib 2 Tabellen zurück
 

df_elements, df_elements_number = define_quadrilateralN4(10,4,0)  # Rufe Funktion auf

df_elements.insert(1, 'Propertie', 0)  # .insert() = Füge neue Spalte ein; Position 1, Name 'Propertie', Wert 0
# df_elements.to_csv(str("elements")+".csv", sep='\t', index=False)  # Speichere als CSV
# df_elements_number.to_csv(str("elements_number")+".csv", sep='\t', index=False)  # Speichere als CSV