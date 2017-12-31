import numpy as np
import random

#from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.642.1549&rep=rep1&type=pdf
class AF:
    def __init__(self, alfa, attacks, supports=[], Amin=0.5):
        self._alfa = alfa
        self._attacks = attacks

        n = len(alfa)
        self.input_layer = np.zeros(n)
        self.output_layer = np.zeros(n)
        self.Amin = Amin

        W = 1/self.Amin * np.arctan(self.Amin)
        self.W = [np.zeros((n, n)), np.zeros((n, n))]
        self.b = np.zeros(n)

        for i in range(n):
            self.W[0][i, i] = np.random.uniform(W, 1)
            self.W[1][i, i] = self.W[0][i, i]

        for a in attacks:
            (i, j) = a
            n = np.sum(attacks[:, 1] == j)

            W = self.W[0][i, i]

            W1 = np.random.uniform(
                -1,
                (2*np.arctan(-self.Amin) + (self.Amin - 1)*W) / ((n + 1)*self.Amin - n + 1)
            )
            self.W[1][i, j] = W1
            self.b[i] = np.random.uniform(
                np.arctan(self.Amin) - self.Amin*W + n*self.Amin*W1,
                np.arctan(-self.Amin) - W + (n - 1 - self.Amin)*W1
            )

        for s in supports:
            (i, j) = s
            n = np.sum(attacks[:, 1] == j)
            W = self.W[0][j]
            W1 = self.W[1][i]
            Ws = np.random.uniform(
                (2*np.arctan(-self.Amin) + (self.Amin-1)*W) / ((n + 1)*self.Amin - n + 1),
                1
            )
            self.W[1][j] = Ws
            self.b[j] = np.random.uniform(
                np.arctan(self.Amin) - n*W1 + W - self.Amin*Ws,
                np.arctan(-self.Amin) - self.Amin*W1 + (n -1)*W1 + self.Amin*W + self.Amin*Ws
            )

    def run(self, inputs):
        hiddens = np.dot(inputs, self.W[0])

        outputs = np.dot(hiddens, self.W[1]) + self.b

        outputs[outputs < -self.Amin] = -1
        outputs[outputs > self.Amin] = 1
        outputs[np.logical_and(outputs >= -self.Amin, outputs <= self.Amin)] = 0

        if (inputs == outputs).all():
            return outputs
        else:
            return self.run(outputs)
