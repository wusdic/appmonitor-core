"""SignatureStore — loads the preset behaviour-signature definitions.

Signatures live in YAML data files, not code, so the "what does this metric
combination mean" library grows by editing data. A signature maps a set of
metric conditions to a human-meaningful activity label + category + severity.

Schema (per signature):
  id:        unique id
  label:     what the entity appears to be doing (shown to operators)
  category:  browse | search | api | transfer | auth | admin | scan |
             tunnel | beacon | recon | maintenance | integration | other
  severity:  info | low | medium | high | critical
  weight:    overall importance multiplier (default 1.0)
  scope:     optional list of business-system ids this applies to ([]=all)
  all:       list of conditions that must ALL hold (AND)
  any:       list of conditions of which at least one must hold (OR)
  none:      list of conditions that must NOT hold
  description: free text
Condition:
  metric: <metric name in the entity snapshot>
  op:     gt|ge|lt|le|eq|ne|between|in|not_in|present
  value:  scalar | [lo,hi] for between | [..] for in/not_in
  soft:   optional band width -> fuzzy 0..1 membership instead of hard 0/1
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


@dataclass
class Signature:
    id: str
    label: str
    category: str = "other"
    severity: str = "info"
    weight: float = 1.0
    scope: List[str] = field(default_factory=list)
    all: List[Dict[str, Any]] = field(default_factory=list)
    any: List[Dict[str, Any]] = field(default_factory=list)
    none: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""


class SignatureStore:
    def __init__(self) -> None:
        self.signatures: List[Signature] = []

    def load_dir(self, path: str) -> int:
        count = 0
        if not os.path.isdir(path):
            return 0
        for fn in sorted(os.listdir(path)):
            if fn.endswith((".yaml", ".yml")):
                count += self.load_file(os.path.join(path, fn))
        return count

    def load_file(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        n = 0
        for item in data:
            self.signatures.append(Signature(
                id=item["id"], label=item.get("label", item["id"]),
                category=item.get("category", "other"),
                severity=item.get("severity", "info"),
                weight=float(item.get("weight", 1.0)),
                scope=item.get("scope", []) or [],
                all=item.get("all", []) or [],
                any=item.get("any", []) or [],
                none=item.get("none", []) or [],
                description=item.get("description", "")))
            n += 1
        return n

    def for_system(self, system: str) -> List[Signature]:
        return [s for s in self.signatures if not s.scope or system in s.scope]

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [s.__dict__ for s in self.signatures]
