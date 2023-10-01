import numpy as np

from marquetry import dataset


class Spiral(dataset.Dataset):
    """Spiral Dataset for Classification.

        The Spiral dataset is a synthetic dataset often used for classification tasks.
        It consists of multiple spiralsn originating from the center and branching outwards in different directions.

        Examples:
            >>> from marquetry.datasets import Spiral
            >>> train_dataset = Spiral(class_num=3, class_data_size=200, random_state=2023)
            >>> test_dataset = Spiral(class_num=3, class_data_size=100, random_state=2023)
            >>> print(len(train_dataset), len(test_dataset))
            600 300
            >>> print(train_dataset[0])
            (array([0.        , 0.        ]), array([1., 0., 0.]))
    """
    def __init__(self, transform=None, target_transform=None,
                 random_state=2023, class_num=3, class_data_size=200):
        self.random_state = random_state
        self.class_num = class_num
        self.class_data_size = class_data_size
        super().__init__(train=True, transform=transform, target_transform=target_transform)

    def _set_data(self):
        np.random.seed(self.random_state)

        class_data_size = self.class_data_size
        data_dim = 2
        class_nums = self.class_num

        x = np.zeros((class_data_size * class_nums, data_dim))
        t = np.zeros((class_data_size * class_nums, class_nums))

        for j in range(class_nums):
            for i in range(class_data_size):
                rate = i / class_data_size
                radius = 1.0 * rate

                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

                index = class_data_size * j + i

                x[index] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
                t[index, j] = 1

        self.source = x
        self.target = t
