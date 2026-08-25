"""Core pipeline stages: data, topic modeling, network construction."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from topiclayers.core.data import Dataset
    from topiclayers.core.network import NetworkCreator
    from topiclayers.core.topic import TopicModeler

__all__ = ["Dataset", "NetworkCreator", "TopicModeler"]


def __getattr__(name: str):
    if name == "Dataset":
        from topiclayers.core.data import Dataset

        return Dataset
    if name == "NetworkCreator":
        from topiclayers.core.network import NetworkCreator

        return NetworkCreator
    if name == "TopicModeler":
        from topiclayers.core.topic import TopicModeler

        return TopicModeler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
