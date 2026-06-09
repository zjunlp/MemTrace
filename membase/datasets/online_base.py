from pydantic import BaseModel
from ..model_types.dataset import (
    MemoryDataset,
    Message,
    Session,
)
from ..layers.base import MemBaseLayer
from ..model_types.evaluation import OnlineEvalResult


class OnlineEvalEnv(BaseModel):
    """Base evaluation environment configuration.

    This class is intentionally empty.  It serves purely as a type marker.  
    Concrete dataset classes should subclass this and define all 
    dataset-specific parameters.
    """


class OnlineMemBaseDataset(MemoryDataset):
    """Base class for datasets that support online evaluation.

    In online evaluation, memory construction and evaluation 
    are interleaved. Subclasses must implement `online_evaluate`.  
    All evaluation logic, including task identification, memory retrieval, 
    memory utilization, memory update, and evaluation, 
    is the subclass's responsibility.
    """

    @classmethod
    def online_evaluate(
        cls,
        messages: Message | list[Message] | Session,
        layer: MemBaseLayer,
        env: OnlineEvalEnv,
    ) -> list[OnlineEvalResult]:
        """Perform end-to-end online evaluation for task message(s).

        Args:
            messages (`Message | list[Message] | Session`):
                The task message(s) to evaluate. If a batch of task
                messages is provided, the evaluation will be performed 
                parallelly. 
            layer (`MemBaseLayer`):
                The live memory layer to retrieve from and update.
            env (`OnlineEvalEnv`):
                Dataset-specific evaluation environment configuration.

        Returns:
            `list[OnlineEvalResult]`:
                Per-task evaluation results. Each contains the full
                rollout trace and computed metrics.
        """
        raise NotImplementedError(
            "This method must be implemented by the subclass."
        )
