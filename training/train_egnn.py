"""Train the PrOMet EGNN.

    python training/train_egnn.py fit --config training/configs/egnn.yaml
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightning.pytorch.cli import LightningCLI  # noqa: E402

from lightning_module import LigMetDataModule, LigMetModel  # noqa: E402

if __name__ == "__main__":
    LightningCLI(LigMetModel, LigMetDataModule, save_config_kwargs={"overwrite": True})
