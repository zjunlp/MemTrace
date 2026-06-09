from collections import OrderedDict
from ..utils._lazy_mapping import _LazyMapping


_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        ("MemBase", "MemBaseDataset"),
        ("LongMemEval", "LongMemEval"),
        ("LoCoMo", "LoCoMo"),
        ("RealMem", "RealMem"),
    ]
)

_MODULE_MAPPING: OrderedDict[str, str] = OrderedDict(
    [
        ("MemBase", "base"),
        ("LongMemEval", "longmemeval"),
        ("LoCoMo", "locomo"),
        ("RealMem", "realmem"),
    ]
)

DATASET_MAPPING = _LazyMapping(
    mapping=_MAPPING_NAMES,
    module_mapping=_MODULE_MAPPING,
    package=__package__,
)

_ENV_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        ("RealMem", "RealMemEvalEnv"),
    ]
)

_ENV_MODULE_MAPPING: OrderedDict[str, str] = OrderedDict(
    [
        ("RealMem", "realmem"),
    ]
)

ONLINE_EVAL_ENV_MAPPING = _LazyMapping(
    mapping=_ENV_MAPPING_NAMES,
    module_mapping=_ENV_MODULE_MAPPING,
    package=__package__,
)

__all__ = ["DATASET_MAPPING", "ONLINE_EVAL_ENV_MAPPING"]
