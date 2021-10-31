from pydantic import BaseModel
from typing import List


class InitRbf(BaseModel):
    num_inputs: int
    radial_centers: int


class FitRbf(BaseModel):
    inputs: List[List[float]]
    outputs: List[float]
    tolerance: float
    epochs: int


class EvalRbf(BaseModel):
    inputs: List[float]


class RadialCenterResponse(BaseModel):
    num_inputs: int
    centroids: List[float]


class NodeResponse(BaseModel):
    num_inputs: int
    weights: List[float]
    bias: float


class RbfResponse(BaseModel):
    num_inputs: int
    radial_centers: List[RadialCenterResponse]
    output_node: NodeResponse
    errors: List[float]
