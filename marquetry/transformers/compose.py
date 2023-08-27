class Compose(object):
    def __init__(self, transforms=None):
        self.transforms = transforms if len(transforms) != 0 else []

    def __call__(self, data):
        if not self.transforms:
            return data

        for transform_func in self.transforms:
            data = transform_func(data)

        return data
