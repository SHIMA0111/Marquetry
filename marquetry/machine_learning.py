import numpy as np

import marquetry


try:
    import cupy
    allow_array = (np.ndarray, cupy.ndarray)
except ImportError:
    cupy = np
    allow_array = (np.ndarray,)


class MachineLearning(object):
    """Base class for machine learning models.

        This class defines the base structure for machine learning models and provides methods for fitting,
        predicting, scoring, and managing model parameters.

        Subclasses must implement the abstract methods _fit_method, _predict_method, save_params, and load_params.
    """

    def fit(self, x, t):
        """Fit the model to the training data.

            Args:
                x (:class:`marquetry.Variable` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                    The input features for training.
                t (:class:`marquetry.Variable` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                    The target values for training.

            Returns:
                None

        """

        inputs = (marquetry.as_variable(x), marquetry.as_variable(t))
        x, t = inputs[0].data, inputs[1].data

        if len(x) != t.size and x.shape[0] == t.shape[0]:
            t = t.argmax(axis=1)
        elif len(x) != t.size:
            raise ValueError("x and t should have the same length, "
                             "but got x: {}, t: {}".format(len(x), len(t)))

        self._fit_method(x, t)

    def predict(self, x):
        """Make predictions using the trained model.

            Args:
                x (:class:`marquetry.Variable` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                    The input features for prediction.

            Returns:
            :class:`marquetry.Variable`: The model's predictions.

        """

        x = marquetry.as_variable(x).data
        y = self._predict_method(x)
        y = marquetry.as_variable(y)

        return y

    def score(self, x, t, evaluator):
        """Evaluate the model's performance using an evaluator function.

            Args:
                x (:class:`marquetry.Variable` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                    The input features for evaluation.
                t (:class:`marquetry.Variable` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                    The target values for evaluation.
                evaluator (callable): A function or callable object that computes the model's score.

            Returns:
                float: The model's score as computed by the evaluator.

            Raises:
            NotImplementedError: This method is abstract and must be implemented by subclasses.
        """

        predict_result = self.predict(x)
        score = evaluator(predict_result, t)

        return score

    def _fit_method(self, x, t):
        raise NotImplementedError()

    def _predict_method(self, x):
        raise NotImplementedError()

    def save_params(self, path):
        """Method for saving model parameters to a file.

            Args:
                path (str): The path to the file where model parameters should be saved.

            Returns:
                None

        """

        raise NotImplementedError()

    def load_params(self, path):
        """Method for loading model parameters from a file.

            Args:
                path (str): The path to the file from which model parameters should be loaded.

            Returns:
                None

        """

        raise NotImplementedError()
