import gzip
import pickle
import numpy as np 

def load(caminho):
    """
    :(str) caminho: caminho para o arquivo MNIST com as imagens a serem classificadas.
    :(tuple) output: tupla na forma (dados de treino, dados de validação, dados de teste).

    Esta função retorna os dados na forma de três listas (dados, validação). A primeira
    para treino, a segunda para validação e a terceira para testes.
    """

    arquivo = gzip.open(caminho)
    dados_treino, dados_validação, dados_teste = pickle.load(arquivo, encoding="latin1")

    arquivo.close()

    dados_treino_final = []

    for j in range(len(dados_treino[0])):

        dados_treino_final.append((np.reshape(dados_treino[0][j], (784,1)), vetorizar_resultado(dados_treino[1][j])))

    dados_validação_final = []

    for j in range(len(dados_validação[0])):

        dados_validação_final.append((np.reshape(dados_validação[0][j], (784,1)), vetorizar_resultado(dados_validação[1][j])))

    dados_teste_final = []

    for j in range(len(dados_teste[0])):

        dados_teste_final.append((np.reshape(dados_teste[0][j], (784,1)), vetorizar_resultado(dados_teste[1][j])))

    return dados_treino_final, dados_validação_final, dados_teste_final

def vetorizar_resultado(j):
    """
    :(int) j: classificação.
    :(array) output: vetor de dimensão 10 com o valor 1.0 no item j.

    Transforma a classificação dos conjuntos de dados em um vetor de dimensão 10.
    """

    vetor = np.zeros((10,1))
    vetor[j] = 1.0

    return vetor