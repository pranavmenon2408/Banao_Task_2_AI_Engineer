"""Named models: the shipped default in artifacts/, user-trained ones in artifacts/models/<name>/."""

import json
import re
import shutil
import threading

import lightgbm as lgb

from .config import ARTIFACT_DIR

DEFAULT = "default"
USER_DIR = ARTIFACT_DIR / "models"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def validate_name(name):
    if not NAME_RE.match(name or ""):
        raise ValueError("model name must be 1-40 letters, digits, '_' or '-'")
    if name == DEFAULT:
        raise ValueError("'default' is reserved for the shipped model")


def model_dir(name):
    return ARTIFACT_DIR if name == DEFAULT else USER_DIR / name


class Registry:
    def __init__(self):
        self._cache, self._lock = {}, threading.Lock()

    def names(self):
        user = sorted(p.name for p in USER_DIR.glob("*") if (p / "model.txt").exists()) if USER_DIR.exists() else []
        return [DEFAULT, *user]

    def meta(self, name):
        return json.loads((model_dir(name) / "meta.json").read_text())

    def list(self):
        out = []
        for n in self.names():
            m = self.meta(n)
            out.append(
                {
                    "name": n,
                    "trained_on": m.get("trained_on"),
                    "subset": m.get("subset", "all labelled claims"),
                    "validation": m.get("validation"),
                    "created": m.get("created"),
                }
            )
        return out

    def get(self, name):
        with self._lock:
            if name not in self._cache:
                d = model_dir(name)
                if not (d / "model.txt").exists():
                    raise KeyError(f"no model named {name!r}; available: {', '.join(self.names())}")
                self._cache[name] = (lgb.Booster(model_file=str(d / "model.txt")), self.meta(name))
            return self._cache[name]

    def save(self, name, booster, meta):
        d = model_dir(name)
        tmp = d.with_name(d.name + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(tmp / "model.txt"))
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        with self._lock:
            if d.exists():
                shutil.rmtree(d)
            tmp.rename(d)
            self._cache.pop(name, None)

    def delete(self, name):
        validate_name(name)
        with self._lock:
            shutil.rmtree(model_dir(name), ignore_errors=True)
            self._cache.pop(name, None)
