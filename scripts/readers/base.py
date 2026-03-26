"""Base types shared across all readers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

try:
    import anndata as ad
except ImportError:
    ad = None  # type: ignore


@dataclass
class ReaderResult:
    """Container returned by every reader on success."""

    adata: "ad.AnnData"
    source_meta: dict = field(default_factory=dict)


class DecisionNeeded(Exception):
    """Raised when a reader encounters an ambiguity requiring user input.

    Parameters
    ----------
    decision_type : str
        Category of the decision (e.g. "gene_column", "layer_choice").
    context : str
        Human-readable description of the ambiguity.
    options : list
        Available choices.
    recommendation : str or None
        The reader's best-guess default, if any.
    reason : str
        Why the reader cannot proceed automatically.
    partial_state : dict
        Serialisable state so the reader can resume after the decision.
    """

    def __init__(
        self,
        decision_type: str,
        context: str,
        options: List[Any],
        recommendation: Optional[str] = None,
        reason: str = "",
        partial_state: Optional[dict] = None,
    ) -> None:
        self.decision_type = decision_type
        self.context = context
        self.options = options
        self.recommendation = recommendation
        self.reason = reason
        self.partial_state = partial_state or {}
        super().__init__(context)

    def to_json(self) -> str:
        """Return a structured JSON representation of this decision request."""
        return json.dumps(
            {
                "decision_type": self.decision_type,
                "context": self.context,
                "options": self.options,
                "recommendation": self.recommendation,
                "reason": self.reason,
                "partial_state": self.partial_state,
            },
            indent=2,
        )
