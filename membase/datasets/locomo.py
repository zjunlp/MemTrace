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


class LoCoMo(MemBaseDataset):
    """Dataset wrapper for LoCoMo."""

    @classmethod
    def read_raw_data(cls, path: str) -> Self:
        category_id_to_type = {
            1: "multi-hop",
            2: "temporal",
            3: "open-domain",
            4: "single-hop",
            5: "adversarial",
        }

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectories = []
        qa_pair_lists = []

        for sample_idx, sample in enumerate(data, start=1):
            trajectory = sample["conversation"]
            speaker_a = trajectory["speaker_a"]
            speaker_b = trajectory["speaker_b"]

            session_summaries = sample["session_summary"]
            session_observations = sample["observation"]

            sessions = []

            for s_idx in range(1, len(session_summaries) + 1):
                session_summary = session_summaries[f"session_{s_idx}_summary"]
                session_observation = session_observations[f"session_{s_idx}_observation"]
                session = trajectory[f"session_{s_idx}"]
                session_ts = trajectory[f"session_{s_idx}_date_time"]
                # Parse LoCoMo-style datetime string like ``'1:56 pm on 8 May, 2023'``. 
                iso_ts = datetime.strptime(
                    session_ts, 
                    "%I:%M %p on %d %B, %Y"
                ).isoformat()

                messages = [] 

                for msg in session:
                    speaker = msg["speaker"]
                    text = msg["text"]
                    if speaker == speaker_a:
                        speaker_tag = "speaker_a"
                    elif speaker == speaker_b:
                        speaker_tag = "speaker_b"
                    else:
                        raise ValueError(f"The speaker '{speaker}' is unknown.")

                    msg_metadata = {"speaker_tag": speaker_tag}
                    msg_id = f"U{sample_idx}:{msg['dia_id']}"
                    for key, value in msg.items():
                        if key != "speaker" and key != "text":
                            msg_metadata[key] = value

                    messages.append(
                        Message(
                            id=msg_id,
                            name=speaker,
                            content=text,
                            role="user",
                            timestamp=iso_ts,
                            metadata=msg_metadata,
                        )
                    )

                sessions.append(
                    Session(
                        id=f"locomo-{sample_idx}-session-{s_idx}",
                        messages=messages,
                        metadata={
                            "session_summary": session_summary,
                            "session_observation": session_observation,
                        },
                    )
                )

            trajectories.append(
                Trajectory(
                    id=f"locomo-{sample_idx}",
                    sessions=sorted(sessions),
                    metadata={
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                    },
                )
            )

            question_ts = sessions[-1].ended_at

            qa_pairs = []
            for q_idx, qa in enumerate(sample["qa"], start=1):
                answer = qa.get("answer")
                if answer is None:
                    answer = "Sorry, I don't know the answer of this question."
                if isinstance(answer, int):
                    answer = str(answer)

                category = qa["category"]
                category_type = category_id_to_type[category]
                qa_metadata = {
                    "question_type": category_type,
                    "category_id": category,
                    "evidence": [
                        f"U{sample_idx}:{evidence}"
                        for evidence in qa["evidence"]
                    ],
                    "speaker_names": [speaker_a, speaker_b],
                }
                if "adversarial_answer" in qa:
                    qa_metadata["adversarial_answer"] = qa["adversarial_answer"]

                qa_pairs.append(
                    QuestionAnswerPair(
                        id=f"locomo-qa-{sample_idx}-{q_idx}",
                        question=qa["question"],
                        golden_answers=[answer],
                        timestamp=question_ts,
                        metadata=qa_metadata,
                    )
                )

            qa_pair_lists.append(qa_pairs)

        return cls(
            trajectories=trajectories,
            qa_pair_lists=qa_pair_lists,
        )

    def _generate_metadata(self) -> dict[str, Any]:
        meta = {
            "name": "LoCoMo",
            "paper": "Evaluating Very Long-Term Conversational Memory of LLM Agents",
            "paper_url": "https://arxiv.org/abs/2402.17753",
            "codebase_url": "https://github.com/snap-research/LoCoMo",
            "homepage": "https://snap-research.github.io/locomo/",
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
        return "locomo-judge"

    @classmethod
    def parse_judge_response(cls, content: str) -> float:
        return float("correct" in content.lower())
