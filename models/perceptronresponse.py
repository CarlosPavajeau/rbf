from pydantic import BaseModel
from typing import List


class InitPerceptron(BaseModel):
    num_inputs: int
    num_layers: int
    nodes_per_layer: List[int]
    activation_functions_names: List[str]


class FitPerceptron(BaseModel):
    inputs: List[List[float]]
    outputs: List[List[float]]
    learning_rate: float
    tolerance: float
    epochs: int


class EvalPerceptron(BaseModel):
    inputs: List[float]


class NodeResponse(BaseModel):
    num_inputs: int
    weights: List[float]
    bias = float


class LayerResponse(BaseModel):
    num_inputs: int
    nodes: List[NodeResponse]
    activation_function_name: str


class PerceptronResponse(BaseModel):
    num_inputs: int
    num_layers: int
    layers: List[LayerResponse]
    activation_function_names: List[str]
    errors: List[float]
