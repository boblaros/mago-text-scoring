from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.inference.base import BaseModelRunner
from app.inference.runners.torch_sequence import TorchBiLSTMAttentionRunner
from app.inference.runners.transformers import TransformersSequenceClassifierRunner
from app.registry.contracts import RegisteredModel


RunnerBuilder = Callable[[RegisteredModel], BaseModelRunner]
RunnerPredicate = Callable[[RegisteredModel], bool]


@dataclass(slots=True)
class RunnerPlugin:
    name: str
    predicate: RunnerPredicate
    builder: RunnerBuilder


class InferencePluginRegistry:
    def __init__(self) -> None:
        self._plugins: list[RunnerPlugin] = []
        self.register(
            "transformers-sequence-classification",
            lambda model: model.manifest.framework.type == "transformers"
            and model.manifest.framework.task == "sequence-classification",
            TransformersSequenceClassifierRunner,
        )
        self.register(
            "torch-bilstm-attention",
            lambda model: model.manifest.framework.type == "pytorch"
            and (model.manifest.framework.architecture or "").lower() == "bilstm-attention",
            TorchBiLSTMAttentionRunner,
        )

    def register(
        self,
        name: str,
        predicate: RunnerPredicate,
        builder: RunnerBuilder,
    ) -> None:
        self._plugins.append(RunnerPlugin(name=name, predicate=predicate, builder=builder))

    def create(self, model: RegisteredModel) -> BaseModelRunner:
        for plugin in self._plugins:
            if plugin.predicate(model):
                return plugin.builder(model)
        framework = model.manifest.framework
        raise ValueError(
            f"No inference plugin registered for framework='{framework.type}' "
            f"task='{framework.task}' architecture='{framework.architecture}'."
        )

    def supports(self, model: RegisteredModel) -> bool:
        return any(plugin.predicate(model) for plugin in self._plugins)
