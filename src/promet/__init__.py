"""PrOMet: structure-based prediction of protein metal-binding sites and metal types."""
__version__ = "1.0.0"

from promet.constants import metals as METALS  # noqa: E402

__all__ = ["METALS", "PrOMet"]


def __getattr__(name):
    if name == "PrOMet":
        from promet.predict import PrOMet
        return PrOMet
    raise AttributeError(name)
