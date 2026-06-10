import numpy as np

from marquetry.machine_learning import MachineLearning


class SVM(MachineLearning):
    """Linear Support Vector Machine for binary classification.

        The model is trained by projected gradient ascent on the dual problem.
        The equality constraint of the dual is omitted for simplicity and the bias
        is recovered afterward by averaging over the margin support vectors,
        which is a common simplification for educational linear SVMs.

        Args:
            c (float or None): The soft margin parameter which limits the dual
                coefficients to ``0 <= alpha <= c``. If ``None``, the model is
                trained as a hard margin SVM (the data should be linearly separable).
            learn_rate (float): The learning rate of the dual gradient ascent.
            epoch (int): The number of optimization steps.
            random_state (int or None): The random seed for the initial dual coefficients.

        Attributes:
            w (:class:`numpy.ndarray`): The weight vector of the separating hyperplane.
            b (float): The bias of the separating hyperplane.

        Note:
            Labels can be either ``{-1, 1}`` or any two distinct values like ``{0, 1}``.
            Predictions are returned using the same label values as the training data.

        Examples:
            >>> model = SVM(c=1.0)
            >>> model.fit(train_x, train_t)
            >>> predictions = model.predict(test_x)
    """

    def __init__(self, c=1.0, learn_rate=0.001, epoch=1000, random_state=2023):
        if c is not None and c <= 0:
            raise ValueError("c should be a positive value or None(hard margin), but got {}".format(c))

        self.c = c
        self.lr = learn_rate
        self.epoch = epoch
        self.random_state = random_state

        self.w = None
        self.b = None
        self.alpha = None
        self.is_trained = False

        self._classes = None

    def _fit_method(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        t = np.asarray(t)

        unique_classes = np.unique(t)
        if len(unique_classes) != 2:
            raise ValueError("SVM supports only binary classification, "
                             "but got {} classes.".format(len(unique_classes)))

        self._classes = unique_classes
        signed_t = np.where(t == unique_classes[1], 1.0, -1.0)

        num_samples = x.shape[0]

        random_gen = np.random.RandomState(self.random_state)
        self.alpha = np.clip(random_gen.normal(loc=0., scale=0.01, size=num_samples), 0., self.c)

        gram = (signed_t[:, np.newaxis] * signed_t[np.newaxis, :]) * (x @ x.T)
        for _ in range(self.epoch):
            grad = np.ones(num_samples) - gram @ self.alpha
            self.alpha += self.lr * grad
            self.alpha = np.clip(self.alpha, 0., self.c)

        support_mask = self.alpha > 1e-10
        if not support_mask.any():
            raise RuntimeError("No support vector was found. "
                               "Please review the learning rate or the epoch number.")

        if self.c is not None:
            margin_mask = support_mask & (self.alpha < self.c - 1e-10)
        else:
            margin_mask = support_mask

        # bias is recovered from on-margin vectors; fall back to all support vectors
        # when every alpha is saturated at c
        if not margin_mask.any():
            margin_mask = support_mask

        self.w = (self.alpha * signed_t)[support_mask] @ x[support_mask]
        self.b = float(np.mean(signed_t[margin_mask] - x[margin_mask] @ self.w))

        self.is_trained = True

    def _predict_method(self, x):
        if not self.is_trained:
            raise RuntimeError("Please train the model before predicting.")

        x = np.asarray(x, dtype=np.float64)
        decision = x @ self.w + self.b

        return np.where(decision > 0, self._classes[1], self._classes[0])

    def decision_function(self, x):
        """Return the signed distance-like score ``x @ w + b`` for each sample."""
        if not self.is_trained:
            raise RuntimeError("Please train the model before predicting.")

        return np.asarray(x, dtype=np.float64) @ self.w + self.b

    def save_params(self, path):
        if not self.is_trained:
            raise RuntimeError("There is no trained parameter to save.")

        np.savez_compressed(path, w=self.w, b=self.b, classes=self._classes)

    def load_params(self, path):
        npz = np.load(path)

        self.w = npz["w"]
        self.b = float(npz["b"])
        self._classes = npz["classes"]
        self.is_trained = True
