"""Configuration files (Section 3.6.5): experiments are defined by files, not by changes to the code.

A configuration is a YAML file.  ``inherits: <file>`` (relative to the file itself) loads another file first
and merges the present one over it, so every experiment file only states what differs from
``configs/base.yaml``.  Single values can be overridden on the command line with ``--set a.b=value``.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (lists and scalars are replaced).

    A section that contains ``_replace: true`` replaces the section of the parent instead of being merged
    into it (used, for example, to list a different set of ``models``).
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("_replace"):
            out[key] = {k: copy.deepcopy(v) for k, v in value.items() if k != "_replace"}
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_file(path: Path, seen: tuple = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular 'inherits' in {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("inherits", None)
    if parent:
        data = deep_merge(_load_file(path.parent / parent, seen + (path,)), data)
    return data


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply ``a.b.c=value`` overrides; the value is parsed as YAML (numbers, lists, true/false, null)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override {key}: {part} is not a section")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    """Load a configuration file with its parents and overrides; relative paths are resolved against the
    project root (the directory that holds ``configs/``, or the current directory)."""
    path = Path(path)
    cfg = apply_overrides(_load_file(path), overrides)
    cfg["_file"] = str(path)
    root = project_root(path)
    paths = cfg.setdefault("paths", {})
    for key, default in (("data", "data/orlib"), ("artifacts", "artifacts"), ("results", "results")):
        p = Path(paths.get(key, default))
        paths[key] = str(p if p.is_absolute() else root / p)
    return cfg


def project_root(config_path: Path) -> Path:
    """The directory above ``configs/`` for a file inside it, otherwise the current directory."""
    resolved = Path(config_path).resolve()
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent
    return Path.cwd()


#: sections that do not change the results and are therefore left out of the identifier of a run
_NOT_HASHED = ("paths", "_file", "workers", "description", "pilot")


def config_hash(cfg: dict) -> str:
    """Short, stable identifier of the settings that define the results of an experiment."""
    core = {k: v for k, v in cfg.items() if k not in _NOT_HASHED}
    if isinstance(core.get("training"), dict) and "verbose" in core["training"]:   # logging only
        core["training"] = {k: v for k, v in core["training"].items() if k != "verbose"}
    blob = json.dumps(core, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def dump_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, fh, sort_keys=False)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """``get(cfg, "entropy.calibration.tolerance", 0.01)``."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
