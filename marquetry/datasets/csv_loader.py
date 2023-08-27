from marquetry import dataset


class CsvLoader(dataset.Dataset):
    def __init__(self):
        super().__init__()

    def _set_data(self, *args, **kwargs):
        raise NotImplementedError()
