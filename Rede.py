"""
Este código contempla a resolução de um problema primordial no aprendizado de máquina - a classificação em software
de algarismos escritos à mão. A base para esta implementação está contida no livro "Neural Networks and Deep 
Learning" de Michael A. Nielsen e serviu de estudo para a introdução no assunto.
"""

import numpy as np
import random

class Rede:
    """
    Classe que implementa redes neurais e o método de aprendizagem por máximo declive.
    """

    def __init__(self, tamanhos):
        """
        :(list) tamanhos: lista com os tamanhos das camadas da rede. O item 0 representa o 
        tamanho da primeira camada, o item 1 da segunda camada e assim sucessivamente.

        Inicializa as características do desenho da rede.
        """

        self._n_camadas = len(tamanhos)
        self._tamanhos = tamanhos
        self._biases = [np.random.randn(y,1) for y in tamanhos[1:]]
        self._pesos = [np.random.randn(y, x) for x, y in zip(tamanhos[:-1], tamanhos[1:])]


    def resposta(self, a, leitura=None):
        """
        :(array) a: vetor a ser computado pela rede neural.
        :(array) output: vetor resultante quando a é fornecido à rede neural.
        :(array) output: resposta da rede.

        Calcula a resposta da rede neural a certo estimulo.
        """

        for p, b in zip(self._pesos, self._biases):
            
            a = sigmoide(np.dot(p, a) + b)

        if leitura:

            return np.argmax(a)

        else:

            return a

    def SGD(self, dados_treino, epocas, tamanho_mini_batch, eta, dados_teste=None):
        """
        :(list) dados_treino: lista de tuplas na forma (input, classificação). 
        :(int) epocas: número de épocas para o treino.
        :(int) tamanho_mini_batch: tamanho dos subgrupos dos dados de teste.
        :(float) eta: taxa de aprendizagem.
        :(list) dados_teste: lista de inputs para a rede neural.

        Método responsável pela implementação do método de máximo declive para aprendizagem da rede.
        """

        dados_treino = list(dados_treino)
        n_dados_treino = len(dados_treino)

        if dados_teste:

            dados_teste = list(dados_teste)
            n_dados_teste = len(dados_teste)

        for j in range(epocas):

            random.shuffle(dados_treino)
            mini_batches = [dados_treino[k:k+tamanho_mini_batch] for k in range(0, n_dados_treino, tamanho_mini_batch)]

            for mini_batch in mini_batches:

                self.atualizar_mini_batch(mini_batch, eta)

            if dados_teste:

                print("Época {}: {} / {}.".format(j, self.avaliar(dados_teste), n_dados_teste))

            else: 

                print("Época {} completa.".format(j))


    def atualizar_mini_batch(self, mini_batch, eta):
        """
        :(list) mini_batch: subgrupo da lista de tuplas na forma (input, classificação).
        :(float) eta: taxa de aprendizagem.

        Atualiza os pesos e biases da rede aplicando o método do máximo declive através
        do algoritmo de backpropagation para cada mini_batch.
        """

        nabla_p = [np.zeros(p.shape) for p in self._pesos]
        nabla_b = [np.zeros(b.shape) for b in self._biases]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_p = self.backprop(x, y)

            nabla_p = [np + dnp for np, dnp in zip(nabla_p, delta_nabla_p)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self._pesos = [p-(eta/len(mini_batch))*np for p, np in zip(self._pesos, nabla_p)]
        self._biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self._biases, nabla_b)]


    def backprop(self, x, y):
        """
        :(array) x: vetor com os inputs do conjunto de teste.
        :(array) y: vetor com as classificações do conjunto de teste.
        :(tuple) output: retorna uma tupla com os gradientes.

        Implementa o algoritimo de backpropagation para determinar o gradiente da função
        de custo dependendo dos pesos ou dos biases.
        """

        nabla_p = [np.zeros(p.shape) for p in self._pesos]
        nabla_b = [np.zeros(b.shape) for b in self._biases]

        ativação = x
        ativações = [x]

        zs = []

        for b, p in zip(self._biases, self._pesos):

            z = np.dot(p, ativação) + b

            zs.append(z)

            ativação = sigmoide(z)
            ativações.append(ativação)

        delta = self.derivada_custo(ativações[-1], y) * dsigmoide(zs[-1])

        nabla_b[-1] = delta
        nabla_p[-1] = np.dot(delta, ativações[-2].transpose())

        for l in range(2, self._n_camadas):

            z = zs[-l]
            
            dsig = dsigmoide(z)
            delta = np.dot(self._pesos[-l+1].transpose(), delta) * dsig
            
            nabla_b[-l] = delta
            nabla_p[-l] = np.dot(delta, ativações[-l-1].transpose())

        return (nabla_b, nabla_p)


    def avaliar(self, dados_teste):
        """
        :(list) dados_teste: lista de inputs para a rede neural.
        :(int) output: quantidade de acertos para os testes.

        Avalia a acurácia da rede.
        """
    
        resultados_teste = [(np.argmax(self.resposta(x)), np.argmax(y)) for (x, y) in dados_teste]
        
        return sum(int(x == y) for (x, y) in resultados_teste)


    def derivada_custo(self, output_ativações, y):
        """
        :(array) output_ativações: vetor com o output de determinada ativação.
        :(array) y: vetor com as classificações do conjunto de teste.
        :(array) output: vetor de derivadas parciais dC_x/da para um output de determinada ativação.  
        """

        return (output_ativações-y)
    

def sigmoide(z):
    """
    :(array) z: vetor a ser computado item a item na sigmoide.
    :(array) output: vetor resultante de z na sigmoide.

    Função que ajusta reais em reais entre 0 e 1.
    """ 

    return 1.0/(1.0+np.exp(-z))

def dsigmoide(z):
    """
    :(array) z: vetor a ser computado item a item na sigmoide'.
    :(array) output: vetor resultante de z na sigmoide'.
    
    Computa o valor da primeira derivada da sigmoide em z.
    """ 

    return sigmoide(z)*(1-sigmoide(z))