"""Saving and loading of trained networks together with the settings that produced them (Section 3.6.5)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .fast import FastAZ, FastLR
from .ffnn import AZNetworks
from .gin import GINPolicyValue


def build_model(kind: str, hparams: dict | None = None):
    """``gin`` builds the GIN of LR, ``ffnn`` the pair of fully connected networks of AZ, ``fast_lr`` and
    ``fast_az`` their fast policy networks for the rollouts."""
    hparams = dict(hparams or {})
    if kind == "gin":
        return GINPolicyValue(**hparams)
    if kind == "ffnn":
        return AZNetworks(**hparams)
    if kind == "fast_lr":
        return FastLR(**hparams)
    if kind == "fast_az":
        return FastAZ(**hparams)
    raise ValueError(f"unknown network kind {kind!r}")


def save_checkpoint(path: str | Path, model, meta: dict[str, Any] | None = None) -> Path:
    """Write the weights, the kind and hyperparameters of the network and free-form metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": model.kind,
        "hparams": dict(model.hparams),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": dict(meta or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path):
    """Return ``(model in evaluation mode, meta)``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = build_model(payload["kind"], payload["hparams"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})


def file_fingerprint(path: str | Path) -> str:
    """Short hash of a file; ties stored temperatures to the checkpoint they were calibrated for."""
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]
