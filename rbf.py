import math
import random
import sys

import numpy as np


class Node:
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        self.bias = random.uniform(-1.0, 1.0)

    def eval(self, inputs: list[float]) -> float:
        assert self.num_inputs == len(inputs)

        inner_prod = sum([weight * inp for weight, inp in zip(self.weights, inputs)]) + self.bias

        return inner_prod


def _thin_plate_spline(x: float) -> float:
    return pow(x, 2) * math.log(x)


class RadialCenter:
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.centroids = []

    def euclidean_distance(self, inputs: list[float]) -> float:
        assert self.num_inputs == len(inputs)

        distance = 0.0
        for i in range(self.num_inputs):
            distance += pow(inputs[i] - self.centroids[i], 2)

        return _thin_plate_spline(distance)

    def init_centroids(self, minimum: float, maximum: float):
        self.centroids = [random.uniform(minimum, maximum) for _ in range(self.num_inputs)]


class Rbf:
    def __init__(self, num_inputs: int, radial_centers: int):
        self.num_inputs = num_inputs
        self.radial_centers = [RadialCenter(self.num_inputs) for _ in range(radial_centers)]
        self.output_node = Node(radial_centers)
        self.errors = []

    def eval(self, inputs: list[float]) -> float:
        euclidean_distances = [radial_center.euclidean_distance(inputs) for index, radial_center in
                               enumerate(self.radial_centers)]

        return self.output_node.eval(euclidean_distances)

    def _set_up_radial_centers(self, inputs: list[list[float]]):
        radial_matrix = np.array(inputs)
        minimum = radial_matrix.min(initial=sys.maxsize)
        maximum = radial_matrix.max(initial=0)

        for index, radial_center in enumerate(self.radial_centers):
            radial_center.init_centroids(minimum, maximum)

    def fit(self, inputs: list[list[float]], outputs: list[float], tolerance: float = 0.1, epochs: int = 30) -> bool:
        self._set_up_radial_centers(inputs)

        for i in range(epochs):
            distances = []
            for _, inp in enumerate(inputs):
                euclidean_distances = [1]
                for _, radial_center in enumerate(self.radial_centers):
                    euclidean_distance = radial_center.euclidean_distance(inp)
                    euclidean_distances.append(euclidean_distance)

                distances.append(euclidean_distances)

            distances = np.array(distances)
            solution = np.linalg.solve(distances, outputs)

            self.output_node.bias = solution[0]
            self.output_node.weights = solution[1:]

            # Eval and calculate error
            predicted_outputs = [self.eval(inp) for _, inp in enumerate(inputs)]
            error = 0.0
            for predicted_output, output in zip(predicted_outputs, outputs):
                error += pow(output - predicted_output, 2)

            self.errors.append(error)
            if error <= tolerance:
                return True

        return False
