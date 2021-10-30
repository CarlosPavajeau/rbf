from pydantic import BaseModel


class InitRbf(BaseModel):
    num_inputs: int
    radial_centers: int


class FitRbf(BaseModel):
    inputs: list[list[float]]
    outputs: list[float]
    tolerance: float
    epochs: int


class EvalRbf(BaseModel):
    inputs: list[float]


class RadialCenter(BaseModel):
    num_inputs: int
    centroids: list[float]


class Node(BaseModel):
    num_inputs: int
    weights: list[float]
    bias: float


class Rbf(BaseModel):
    num_inputs: int
    radial_centers: list[RadialCenter]
    output_node: Node
    errors: list[float]
