from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class PhDProfile:
    title: str
    research_question: str
    hypothesis: str
    shock_start: int
    shock_end: int
    informed_agent_share: float
    resilience_agent_share: float
    cancellation_pressure: float
    order_flow_autocorrelation: float
    research_features: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "PhDProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def shock_intensity(self, step: int) -> float:
        if self.shock_start <= step <= self.shock_end:
            middle = (self.shock_start + self.shock_end) / 2
            half_width = max((self.shock_end - self.shock_start) / 2, 1)
            return max(0.0, 1.0 - abs(step - middle) / half_width)
        return 0.0

