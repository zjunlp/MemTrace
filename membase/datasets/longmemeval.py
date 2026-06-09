import json
from datetime import datetime
from .base import MemBaseDataset
from ..model_types.dataset import (
    Trajectory,
    Session,
    QuestionAnswerPair,
    Message,
)
from typing import Any, Self


class LongMemEval(MemBaseDataset):
    """Dataset wrapper for LongMemEval."""

    @classmethod
    def read_raw_data(cls, path: str) -> Self:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectories = []
        qa_pair_lists = []

        for i, sample in enumerate(data, start=1):
            question_datetime = datetime.strptime(
                sample["question_date"], 
                "%Y/%m/%d (%a) %H:%M"
            )
            answer = sample["answer"]
            # Some answers are integers, which should be converted to strings. 
            if isinstance(answer, int):
                answer = str(answer)

            qa_pair = QuestionAnswerPair(
                id=sample["question_id"],
                question=sample["question"],
                golden_answers=[answer],
                timestamp=question_datetime.isoformat(),
                metadata={
                    "question_type": sample["question_type"],
                    "answer_session_ids": sample["answer_session_ids"],
                },
            )

            sessions, evidence = [], [] 
            for j, raw_session in enumerate(sample["haystack_sessions"], start=1):
                # There are some empty sessions to be skipped.  
                if not raw_session:
                    continue
                session_id = sample["haystack_session_ids"][j - 1]
                iso_ts = datetime.strptime(
                    sample["haystack_dates"][j - 1], 
                    "%Y/%m/%d (%a) %H:%M"
                ).isoformat()

                messages = [] 
                for k, message in enumerate(raw_session, start=1):
                    msg_id = f"longmemeval-message-{i}-{j}-{k}"
                    messages.append(
                        Message(
                            id=msg_id,
                            name=message["role"],
                            role=message["role"],
                            content=message["content"],
                            timestamp=iso_ts,
                            metadata={
                                "has_answer": message.get("has_answer", False),
                            },
                        )
                    )

                    if message.get("has_answer", False):
                        evidence.append(msg_id)

                sessions.append(
                    Session(
                        id=session_id,
                        messages=messages,
                    )
                )

            qa_pair.update_metadata({"evidence": evidence})

            trajectories.append(
                Trajectory(
                    id=f"longmemeval-{sample['question_id']}",
                    sessions=sorted(sessions),
                )
            )
            qa_pair_lists.append([qa_pair])

        return cls(
            trajectories=trajectories,
            qa_pair_lists=qa_pair_lists,
        )

    def _generate_metadata(self) -> dict[str, Any]:
        meta = {
            "name": "LongMemEval",
            "paper": "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory",
            "codebase_url": "https://github.com/xiaowu0162/LongMemEval",
            "size": len(self),
            "total_sessions": 0,
            "total_messages": 0,
            "total_questions": 0,
        }

        question_type_stats = {}
        for trajectory, qa_list in self:
            meta["total_sessions"] += len(trajectory)
            meta["total_messages"] += sum(len(s) for s in trajectory)
            meta["total_questions"] += len(qa_list)
            for qa in qa_list:
                qtype = qa.metadata["question_type"]
                question_type_stats[qtype] = question_type_stats.get(qtype, 0) + 1

        meta["question_type_stats"] = question_type_stats

        n_traj = len(self)
        n_sessions = meta["total_sessions"]
        if n_traj > 0 and n_sessions > 0:
            meta["avg_session_per_trajectory"] = n_sessions / n_traj
            meta["avg_message_per_session"] = meta["total_messages"] / n_sessions
            meta["avg_question_per_trajectory"] = meta["total_questions"] / n_traj
        else:
            meta["avg_session_per_trajectory"] = 0.0
            meta["avg_message_per_session"] = 0.0
            meta["avg_question_per_trajectory"] = 0.0

        return meta

    @classmethod
    def get_judge_template_name(cls, qa_pair: QuestionAnswerPair) -> str:
        qtype = qa_pair.metadata.get("question_type", "normal")
        if qtype == "normal":
            prompt_name = "exact-match"
        elif "_abs" in qa_pair.id:
            prompt_name = "longmemeval-abstention"
        else:
            prompt_name = f"longmemeval-{qtype}"
        return prompt_name
