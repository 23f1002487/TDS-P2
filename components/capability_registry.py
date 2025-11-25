"""Capability Registry Component
Tracks which capabilities/features were used during a quiz chain.
"""
from typing import Any, Dict
from copy import deepcopy
from loguru import logger

class CapabilityRegistry:
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def record(self, key: str, value: Any):
        """Record a capability usage.
        If the key already exists with a different value, convert to a list to preserve all distinct values.
        """
        existing = self._store.get(key)
        if existing is None:
            self._store[key] = value
        else:
            # If same value, ignore duplicates
            if existing == value:
                return
            # Promote to list if necessary
            if not isinstance(existing, list):
                self._store[key] = [existing, value]
            else:
                if value not in existing:
                    existing.append(value)
        logger.debug(f"Capability recorded: {key} -> {value}")

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy snapshot of capabilities used."""
        return deepcopy(self._store)

    def reset(self):
        """Reset registry (useful between quiz chains)."""
        self._store.clear()
        logger.debug("Capability registry reset")
