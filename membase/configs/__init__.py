from collections import OrderedDict
from ..utils._lazy_mapping import _LazyMapping


_MAPPING_NAMES = OrderedDict[str, str](
    [
        ("A-MEM", "AMEMConfig"),
        ("LangMem", "LangMemConfig"),
        ("Long-Context", "LongContextConfig"),
        ("NaiveRAG", "NaiveRAGConfig"),
        ("MemOS", "MemOSConfig"),
        ("EverMemOS", "EverMemOSConfig"),
        ("HippoRAG2", "HippoRAGConfig"),
        ("Mem0", "Mem0Config"),
    ]
)

_MODULE_MAPPING = OrderedDict[str, str](
    [
        ("A-MEM", "amem"),
        ("LangMem", "langmem"),
        ("Long-Context", "long_context"),
        ("NaiveRAG", "naive_rag"),
        ("MemOS", "memos"),
        ("EverMemOS", "evermemos"),
        ("HippoRAG2", "hipporag"),
        ("Mem0", "mem0"),
    ]
)

CONFIG_MAPPING = _LazyMapping(
    mapping=_MAPPING_NAMES,
    module_mapping=_MODULE_MAPPING,
    package=__package__,
)


__all__ = ["CONFIG_MAPPING"]
