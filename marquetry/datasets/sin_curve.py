import numpy as np

from marquetry import dataset


class SinCurve(dataset.Dataset):
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