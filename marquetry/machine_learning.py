class MachineLearning(object):
    def fit(self, x, t):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def score(self, x, t, evaluator):
        predict_result = self.predict(x)
        score = evaluator(predict_result, t)

        return score

    def save_params(self, path):
        raise NotImplementedError()

    def load_params(self, path):
        raise NotImplementedError()
