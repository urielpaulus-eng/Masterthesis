from Woods import wood  #importamos el diccionario con los materiales

class Beam:     # la clase define las propiedades que debe tener el beam

    def __init__(self):

        self.wood = None

    def call_woods (self):

        print(wood[self.wood]["density"]) 
        print(wood[self.wood]["E"])

beam_1 = Beam()
beam_1.wood = "oak"

beam_1.call_woods()


