import numpy as np

from marquetry import dataset


class TrigonometricCurve(dataset.Dataset):
    """Get SinCurve dataset.

        Toy problem that is trigonometric number between 0 and 3π.
        The train dataset is sin curve with small noise, and the test dataset is cos curve.

        The Attributes and Args is following the :class:`marquetry.dataset.Dataset`, please check it.

        Examples:
            >>> data = TrigonometricCurve()
            >>> data.source.shape
            (4999, 1)
            >>> source, target = data[0]
            >>> source.shape
            (1,)
    """
    def _set_data(self):
        num_data = 5000
        dtype = np.float64

        x = np.linspace(0, 3 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)

        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)

        y = y.astype(dtype)
        self.target = y[1:][:, np.newaxis]
        self.source = y[:-1][:, np.newaxis]
