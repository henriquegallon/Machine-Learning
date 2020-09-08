from PIL import Image
import numpy as np

def ver(dados):
    """
    :(array) dados: vetor com a imagem do número.

    Está função cria uma imagem do número no conjunto de dados.
    """

    img = Image.fromarray(np.reshape(dados, (28,28))*255, 'F')
    img = img.resize((1000,1000))
    img.show()
