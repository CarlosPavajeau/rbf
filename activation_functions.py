import math
from collections.abc import Callable


def _tanh(x: float) -> float:
    return math.tanh(x)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _linear(x: float) -> float:
    return x


_ACTIVATION_FUNCTIONS = {
    'tanh': _tanh,
    'linear': _linear,
    'sigmoid': _sigmoid
}


def get_activation_function(name: str) -> Callable[[float], float]:
    return _ACTIVATION_FUNCTIONS.get(name)
