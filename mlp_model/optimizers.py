
import numpy as np
from abc import abstractmethod


class Optimizer:
    """
    Base class for optimizer
    """

    @abstractmethod
    def update_weights(self, layers):
        pass

    @abstractmethod
    def initialize_parameters(self, layers):
        pass


class GradientDescent(Optimizer):
    """
    Implements gradient descent optimizer
    """

    def __init__(self, learning_rate=0.001):
        """
        Initialize optimizer
        :param learning_rate: learning rate of each iteration (type: float)
        """
        self.learning_rate = learning_rate
        self.type = "gradient_descent"

    def initialize_parameters(self, layers):
        return layers

    def update_weights(self, layers):
        """
        Perform update rule
        :param layers: layers of the MLP to update (type: list[Dense()])
        :return: layers with updated weights (type: list[Dense()])
        """
        for i in range(len(layers)):
            layers[i].W = layers[i].W - self.learning_rate * layers[i].dW
            layers[i].b = layers[i].b - self.learning_rate * layers[i].db

        return layers


class Adam(Optimizer):
    """
    Implements Adam optimizer
    """

    def __init__(self, learning_rate=0.001):
        """
        Initialize optimizer
        :param learning_rate: learning rate of each iteration (type: float)
        """
        self.type = "adam"
        self.learning_rate = learning_rate
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.t = 1

    def initialize_parameters(self, layers):
        """
        Initializes momemtum and velocity parameters for each layer of MLP

        :param layers: layers of the MLP (type: list[Dense()])
        :return: layers with initialized parameters (type: list[Dense()])
        """
        for i, layer in enumerate(layers):
            adam = {
                "mW": np.zeros([layer.input_dim, layer.output_dim]),
                "mb": np.zeros([1, layer.output_dim]),
                "vW": np.zeros([layer.input_dim, layer.output_dim]),
                "vb": np.zeros([1, layer.output_dim]),
            }
            layers[i].adam = adam
        return layers

    def update_weights(self, layers: list) -> list:
        """
        Perform update rule
        :param layers: layers of the MLP to update (type: list[Dense()])

        :return: layers with updated weights (type: list[Dense()])
        """

        t = self.t
        for i, layer in enumerate(layers):
            adam = {
                "mW": (self.beta_1 * layer.adam["mW"] + (1 - self.beta_1) * layer.dW),
                "mb": (self.beta_1 * layer.adam["mb"] + (1 - self.beta_1) * layer.db),
                "vW": (
                    self.beta_2 * layer.adam["vW"] + (1 - self.beta_2) * layer.dW ** 2
                ),
                "vb": (
                    self.beta_2 * layer.adam["vb"] + (1 - self.beta_2) * layer.db ** 2
                ),
            }

            layer.adam = adam

            mw_corrected = adam["mW"] / (1 - (self.beta_1 ** t))
            mb_corrected = adam["mb"] / (1 - (self.beta_1 ** t))
            vw_corrected = adam["vW"] / (1 - (self.beta_2 ** t))
            vb_corrected = adam["vb"] / (1 - (self.beta_2 ** t))

            layer.W = layer.W - (
                self.learning_rate
                * mw_corrected
                / (np.sqrt(vw_corrected) + self.epsilon)
            )
            layer.b = layer.b - (
                self.learning_rate
                * mb_corrected
                / (np.sqrt(vb_corrected) + self.epsilon)
            )

            layers[i] = layer
        self.t = t + 1
        return layers
