class Compose(object):
    """A utility class for composing multiple data transformation functions in a pipeline.

        Args:
            transforms (list of callable functions):
                A list of callable data transformation functions.

        Examples:
            >>> compose = Compose([
            >>>     marquetry.transformers.Normalize(),
            >>>     marquetry.transformers.Flatten()
            >>> ])
            >>> dataset = marquetry.datasets.MNIST(transform=compose)
    """

    def __init__(self, transforms: list = None):
        self.transforms = transforms if len(transforms) != 0 else []

    def __call__(self, data):
        if not self.transforms:
            return data

        for transform_func in self.transforms:
            data = transform_func(data)

        return data
