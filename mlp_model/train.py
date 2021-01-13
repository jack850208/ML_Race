
import numpy as np
import pandas as pd

from mlp_model.data_processing import Batcher
from mlp_model.losses import Loss
from mlp_model.optimizers import Optimizer
from mlp_model.optimizers import Adam


default_params = {
    "n_epoch": 10,  # number of epochs
    "batch_size": 128,  # batch size
    "n_stopping_rounds": 10,  # N of consecutive epochs without improvement for early-stopping
    "learning_rate": 0.0001,  # learning rate
    "reg_lambda": 0.01,  # regularization factor for gradients
    "verbose": True,  # flag to plot train  results
    "print_rate": 5,  # print train results every print_rate epochs
    "early_stopping": False,  # flag to use early-stopping
}


class ModelTrain:
    def __init__(self, params=None):
        self._train_params = default_params

        self._update_params(params)
        self._batcher = None
        self._optimizer = None

    def train(
        self,
        model,
        loss: Loss,
        train_data: list,
        optimizer: Optimizer = Adam(),
        dev_data: list = None,
        params: dict = None,
    ):
        """
        Run several train steps

        :param model: model to uptimize on (type: Model)
        :param loss: loss function object (type: Loss)
        :param train_data: train dataset containing x,y pair (type: list[np.array])
        :param optimizer: optimizer to use in train (type: Optimizer)
        :param dev_data: train dataset containing x_dev,y_dev pair (type: list[np.array])
        :param params: train parameters (type: dict)
        """

        self._update_params(params)
        self._optimizer = optimizer
        model.layers = self._optimizer.initialize_parameters(model.layers)

        if self._batcher is None:
            self._batcher = Batcher(train_data, self._train_params["batch_size"])

        epoch = 1
        best_loss = 1e14
        early_stopping_counter = 0
        verbose = self._train_params["verbose"]

        # Start train
        model.train_log = []
        model.dev_log = []
        train_loss = []
        while (
            epoch <= self._train_params["n_epoch"]
            and early_stopping_counter < self._train_params["n_stopping_rounds"]
        ):
            self._batcher.reset()

            # Train on batches
            for batch_i in range(self._batcher.n_batches):
                x_batch, y_batch = self._batcher.next()
                self._train_step(
                    model, x_batch, y_batch, loss, self._train_params["reg_lambda"]
                )
                loss_i = self._compute_loss(model.layers[-1].A, y_batch, loss)
                train_loss.append(loss_i)
                model.train_log.append(np.array([epoch, batch_i, loss_i]))

            # Evaluate dev data
            if dev_data is not None:
                x_dev, y_dev = dev_data
                dev_pred = model.predict(x_dev)
                dev_loss = self._compute_loss(dev_pred, y_dev, loss)
                model.dev_log.append(np.array([epoch, dev_loss]))

                if best_loss > dev_loss:
                    early_stopping_counter = 0
                    best_loss = dev_loss
                else:
                    early_stopping_counter += 1

                if verbose and (epoch % self._train_params["print_rate"] == 0):
                    print(
                        f"epoch: {epoch} | train_loss: {np.mean(train_loss)} |  dev_loss: {dev_loss}"
                    )

            else:
                if verbose and (epoch % self._train_params["print_rate"] == 0):
                    print(f"epoch: {epoch} | train_loss: {np.mean(train_loss)}")

            epoch += 1

        model.train_log = np.vstack(model.train_log)
        model.train_log = pd.DataFrame(
            model.train_log, columns=["epoch", "iter", "loss"]
        )
        if dev_data is not None:
            model.dev_log = np.vstack(model.dev_log)
            model.dev_log = pd.DataFrame(model.dev_log, columns=["epoch", "loss"])

    def _train_step(
        self, model, x: np.array, y: np.array, loss: Loss, reg_lambda: float = 0.01
    ):
        """
        Performs a complete train step of the network
        (1) Forward pass: computes Z and A for each layer
        (2) Back propagation: computes gradients for each layer
        (3) Update weights: call optimizer to perform update rule

        :param x: input matrix to the network (type: np.array)
        :param y: target vector (type: np.array)
        """

        # Forward propagation
        _ = model.forward_prop(x)

        # Back propagation
        model.back_prop(x, y, loss, reg_lambda=reg_lambda)

        # Update rule
        model.layers = self._optimizer.update_weights(model.layers)

    @staticmethod
    def _compute_loss(actual: np.array, prediction: np.array, loss: Loss) -> np.array:
        """
        Computes loss between prediction and target

        :param actual: target vector (type: np.array)
        :param prediction: predictions vector (type: np.array)
        :param loss: loss function ibject (type: Loss)

        :return: average loss (type: float)
        """

        current_loss = loss.forward(actual, prediction)
        current_loss = np.mean(current_loss)
        return current_loss

    def _update_params(self, params: dict):
        if params is not None:
            self._train_params.update(params)
