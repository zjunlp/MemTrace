from __future__ import annotations

from dataclasses import asdict
import re
from typing import Iterable

try:
    from .models import GrepMatch, OperationBlock
except ImportError:  # pragma: no cover - supports running this folder as scripts.
    from models import GrepMatch, OperationBlock


_SECTION_LABELS = (
    "Inputs:",
    "Input variables:",
    "Intermediate:",
    "Intermediate variables:",
    "Outputs:",
    "Output variables:",
)


class TraceIndex:
    def __init__(self, flattened_trace_text: str) -> None:
        self.flattened_trace_text = (flattened_trace_text or "").strip()
        self.blocks = self._parse_blocks(self.flattened_trace_text)
        self.op_by_id = {block.op_id: block for block in self.blocks}

    def _split_operation_blocks(self, text: str) -> list[tuple[str, int, int]]:
        if not text:
            return []
        if "### Operation " not in text:
            return [(text, 0, len(text))]

        starts = [match.start() for match in re.finditer(r"(?=### Operation\s+)", text)]
        blocks = []
        if starts[0] > 0 and text[: starts[0]].strip():
            blocks.append(self._trim_block(text, 0, starts[0]))
        for idx, start in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
            blocks.append(self._trim_block(text, start, end))
        return [block for block in blocks if block[0]]

    def _trim_block(self, text: str, start: int, end: int) -> tuple[str, int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return text[start:end], start, end

    def _parse_blocks(self, text: str) -> list[OperationBlock]:
        blocks = []
        for order, (raw, start, end) in enumerate(self._split_operation_blocks(text)):
            op_id = self._parse_op_id(raw, order)
            blocks.append(
                OperationBlock(
                    op_id=op_id,
                    order=order,
                    char_start=start + 1,
                    char_end=end,
                    created_at=self._parse_field(raw, "created_at"),
                    op_name=self._parse_field(raw, "op_name"),
                    category=self._parse_field(raw, "category"),
                    comment=self._parse_field(raw, "comment"),
                    input_text=self._parse_section(raw, ("Inputs:", "Input variables:")),
                    intermediate_text=self._parse_section(raw, ("Intermediate:", "Intermediate variables:")),
                    output_text=self._parse_section(raw, ("Outputs:", "Output variables:")),
                    full_text=raw,
                )
            )
        return blocks

    def _parse_op_id(self, text: str, order: int) -> str:
        match = re.search(r"### Operation\s+(\S+)", text)
        if match:
            return match.group(1).strip()
        match = re.search(r"\bop-[a-zA-Z0-9_-]+\b", text)
        return match.group(0) if match else f"UNKNOWN_OP_{order}"

    def _parse_field(self, text: str, field: str) -> str:
        match = re.search(rf"^{re.escape(field)}\s*:\s*(.*)$", text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
        xml_match = re.search(
            rf"<{re.escape(field)}>(.*?)</{re.escape(field)}>",
            text,
            flags=re.DOTALL,
        )
        return xml_match.group(1).strip() if xml_match else ""

    def _parse_section(self, text: str, start_labels: tuple[str, ...]) -> str:
        starts = [(text.find(label), label) for label in start_labels if text.find(label) != -1]
        if not starts:
            return ""
        start, label = min(starts, key=lambda item: item[0])
        body_start = start + len(label)
        body_end = len(text)
        for end_label in _SECTION_LABELS:
            pos = text.find(end_label, body_start)
            if pos != -1:
                body_end = min(body_end, pos)
        return text[body_start:body_end].strip()

    def get_op(self, op_id: str) -> OperationBlock | None:
        return self.op_by_id.get(op_id)

    def grep(
        self,
        pattern: str,
        fields: Iterable[str] = ("input", "intermediate", "output", "full"),
        regex: bool = False,
        case_sensitive: bool = False,
        max_matches: int = 50,
        surrounding_chars: int = 120,
    ) -> list[GrepMatch]:
        if not pattern:
            return []
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern if regex else re.escape(pattern), flags=flags)
        except re.error:
            compiled = re.compile(re.escape(pattern), flags=flags)

        matches: list[GrepMatch] = []
        for block in self.blocks:
            section_map = {
                "input": block.input_text,
                "intermediate": block.intermediate_text,
                "output": block.output_text,
                "full": block.full_text,
            }
            for section in fields:
                haystack = section_map.get(section, "")
                section_base = 0 if section == "full" else block.full_text.find(haystack)
                if section_base < 0:
                    section_base = 0
                for match in compiled.finditer(haystack):
                    start, end = match.span()
                    left = max(0, start - surrounding_chars)
                    right = min(len(haystack), end + surrounding_chars)
                    snippet = haystack[left:right].replace("\x00", "")
                    matches.append(
                        GrepMatch(
                            op_id=block.op_id,
                            order=block.order,
                            created_at=block.created_at,
                            section=section,
                            match_text=match.group(0),
                            snippet=snippet,
                            start=start,
                            end=end,
                            global_start=block.char_start + section_base + start,
                            global_end=block.char_start + section_base + end - 1,
                        )
                    )
                    if len(matches) >= max_matches:
                        return matches
        return matches


def block_preview(block: OperationBlock) -> dict[str, str | int]:
    return {
        "op_id": block.op_id,
        "order": block.order,
        "char_start": block.char_start,
        "char_end": block.char_end,
        "created_at": block.created_at,
        "op_name": block.op_name,
        "category": block.category,
        "comment": block.comment,
    }


def match_to_dict(match: GrepMatch) -> dict:
    return asdict(match)