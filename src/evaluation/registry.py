from collections.abc import Callable, Mapping
from typing import Any


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._builders: dict[str, Callable[[Mapping[str, Any]], Any]] = {}

    def register(self, name: str, builder: Callable[[Mapping[str, Any]], Any]) -> None:
        if name in self._builders:
            raise ValueError(f"{self.kind} '{name}' is already registered")
        self._builders[name] = builder

    def build(self, cfg: Mapping[str, Any]) -> Any:
        name = cfg.get("name")
        if not name:
            raise ValueError(f"{self.kind} config requires a 'name'")
        if name not in self._builders:
            available = ", ".join(sorted(self._builders)) or "<none>"
            raise ValueError(
                f"unknown {self.kind} '{name}'. available {self.kind}s: {available}"
            )
        return self._builders[str(name)](cfg)

    def names(self) -> list[str]:
        return sorted(self._builders)
