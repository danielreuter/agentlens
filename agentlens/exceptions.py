class DatasetIndexError(Exception):
    """Raised when accessing an invalid row index"""

    pass


class LabelNotFoundError(Exception):
    """Raised when accessing an undefined Label"""

    pass
