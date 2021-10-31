from pydantic import BaseModel


class InitPerceptron(BaseModel):
    num_inputs: int
    num_layers: int
    nodes_per_layer: list[int]
    activation_functions_names: list[str]


class FitPerceptron(BaseModel):
    inputs: list[list[float]]
    outputs: list[list[float]]
    learning_rate: float
    tolerance: float
    epochs: int


class EvalPerceptron(BaseModel):
    inputs: list[float]


class Node(BaseModel):
    num_inputs: int
    weights: list[float]
    bias = float


class Layer(BaseModel):
    num_inputs: int
    nodes: list[Node]
    activation_function_name: str


class Perceptron(BaseModel):
    num_inputs: int
    num_layers: int
    layers: list[Layer]
    activation_function_names: list[str]
