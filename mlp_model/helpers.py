
from mlp_model import activations, losses, optimizers


def assign_activation(type: str) -> activations.Activation:
    """
    Returns actiavtion function given type

    :param type: type of activation funciton (type: str)

    :return: actiavtion function (type: Activation)
    """

    options = {
        "sigmoid": activations.Sigmoid(),
        "swish": activations.Swish(),
        "relu": activations.Relu(),
        "leaky_relu": activations.LeakyRelu(),
        "tanh": activations.Tanh(),
        "linear": activations.Linear(),
    }

    return options[type]


def assign_loss(type: str) -> losses.Loss:
    """
    Returns loss function given type

    :param type: type of loss funciton (type: str)

    :return: loss function (type: Loss)
    """

    options = {
        "mse": losses.MSE(),
        "mae": losses.MAE(),
        "logloss": losses.Logloss(),
        "quantile": losses.Quantile(),
    }

    return options[type]


def assign_optimizer(type: str) -> optimizers.Optimizer:
    """
    Returns optimizer given type

    :param type: type of optimizer (type: str)

    :return: optimizer object (type: Optimizer )
    """

    options = {
        "gradient_descent": optimizers.GradientDescent(),
        "adam": optimizers.Adam(),
    }

    return options[type]
