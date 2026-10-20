"""Locations of the released model parameters."""
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

RELEASE_TAG = "v1.0.0"
RF_FILENAME = "promet_rf.joblib"
RF_URL = f"https://github.com/qkrgangeun/PrOMet/releases/download/{RELEASE_TAG}/{RF_FILENAME}"
RF_SHA256 = "2d73f65bd583b1fffd2791116a11774a35c9032d93537078a22f21d0f2521836"

EGNN_PATH = Path(__file__).parent / "params" / "promet_egnn.pt"


def weights_dir() -> Path:
    """Directory holding downloaded parameters ($PROMET_HOME, default ~/.cache/promet)."""
    return Path(os.environ.get("PROMET_HOME", Path.home() / ".cache" / "promet"))


def default_rf_path() -> Path:
    return weights_dir() / RF_FILENAME


def sha256sum(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_rf(dest=None, force=False) -> Path:
    """Download the Random Forest parameters (~1.8 GB) from the GitHub release and verify the checksum."""
    dest = Path(dest) if dest else default_rf_path()
    if dest.exists() and not force:
        print(f"Found {dest}; verifying checksum ...")
        if sha256sum(dest) == RF_SHA256:
            print("OK")
            return dest
        print("Checksum mismatch; downloading again.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {RF_URL}\n  -> {dest}")
    with urllib.request.urlopen(RF_URL) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 24)
    if sha256sum(tmp) != RF_SHA256:
        tmp.unlink()
        raise RuntimeError("Checksum mismatch for the downloaded RF parameters.")
    tmp.replace(dest)
    print("Done.")
    return dest
