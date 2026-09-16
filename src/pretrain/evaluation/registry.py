"""Registry mapping evaluation task names to their EvaluationTask subclasses.

Populated via the @register_task decorator on each task's class definition.
EvaluationRunner keys off this registry instead of hardcoding the set of
known tasks, so adding a new multiple-choice benchmark (ARC, PIQA,
WinoGrande, TruthfulQA, ...) only requires a new task file plus one yaml
block under evaluation.tasks - no edits to runner.py's constructor.

Registration is a side effect of importing each task module. Importing any
submodule of the pretrain.evaluation package always executes
pretrain/evaluation/__init__.py first (Python's normal package-init
semantics), and that file imports every task module - so by the time
EvaluationRunner.__init__ runs (well after all imports resolve), the
registry is guaranteed fully populated regardless of which submodule
triggered the import.
"""

from pretrain.evaluation.base import EvaluationTask

TASK_REGISTRY: dict[str, type[EvaluationTask]] = {}


def register_task(name: str):
    """Class decorator registering an EvaluationTask subclass under `name`.

    `name` must match the task's `.name` property and the config key used
    for it under `evaluation.tasks` in yaml (e.g. "mmlu").
    """

    def _decorator(cls: type[EvaluationTask]) -> type[EvaluationTask]:
        if name in TASK_REGISTRY:
            raise ValueError(f"Evaluation task '{name}' already registered")
        TASK_REGISTRY[name] = cls
        return cls

    return _decorator
