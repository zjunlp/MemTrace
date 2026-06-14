from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OperationBlock:
    op_id: str
    order: int
    char_start: int
    char_end: int
    created_at: str = ""
    op_name: str = ""
    category: str = ""
    comment: str = ""
    input_text: str = ""
    intermediate_text: str = ""
    output_text: str = ""
    full_text: str = ""


@dataclass(slots=True)
class GrepMatch:
    op_id: str
    order: int
    created_at: str
    section: str
    match_text: str
    snippet: str
    start: int
    end: int
    global_start: int
    global_end: int