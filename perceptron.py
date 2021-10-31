import random
from collections.abc import Callable

from activation_functions import get_activation_function


class Node:
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        self.bias = random.uniform(-1.0, 1.0)

    def eval(self, inputs: list[float], activation_function: Callable[[float], float]) -> float:
        assert self.num_inputs == len(inputs)

        inner_prod = sum([weight * inp for weight, inp in zip(self.weights, inputs)]) + self.bias

        return activation_function(inner_prod)

    def update_weights(self, input_layer_activation: list[float], learning_rate: float, error: float):
        for index, weight in enumerate(self.weights):
            weight += learning_rate * error * input_layer_activation[index]

        self.bias += learning_rate * error


class Layer:
    def __init__(self, num_inputs: int, num_nodes: int, activation_function_name: str):
        self.num_inputs = num_inputs
        self.nodes = [Node(num_inputs) for _ in range(num_nodes)]
        self.activation_function_name = activation_function_name
        self.activation_function = get_activation_function(activation_function_name)

    def eval(self, inputs: list[float]) -> list[float]:
        output = []

        for _, node in enumerate(self.nodes):
            output.append(node.eval(inputs, self.activation_function))

        return output

    def update_weights(self, input_layer_activation: list[float], learning_rate: float, pattern_error: float):
        for _, node in enumerate(self.nodes):
            node.update_weights(input_layer_activation, learning_rate, pattern_error)

    def update_weights_l(self, input_layer_activation: list[float], learning_rate: float, lineal_errors: list[float]):
        for index, node in enumerate(self.nodes):
            node.update_weights(input_layer_activation, learning_rate, lineal_errors[index])


class Perceptron:
    def __init__(self, num_inputs: int, num_layers: int, nodes_per_layer: list[int],
                 activation_functions_names: list[str]):
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.errors = []
        self.all_layers_activations = []

        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(Layer(num_inputs, nodes_per_layer[i], activation_functions_names[i]))
                continue

            self.layers.append(Layer(nodes_per_layer[i - 1], nodes_per_layer[i], activation_functions_names[i]))

    def eval(self, inputs: list[float]) -> list[float]:
        output = []
        temp_in = inputs
        self.all_layers_activations.append(inputs)

        for index, layer in enumerate(self.layers):
            output = layer.eval(temp_in)
            self.all_layers_activations.append(output)
            temp_in = output

        return output

    def _update_weights(self, learning_rate: float, pattern_error: float, lineal_errors: list[float]):
        for index, layer in enumerate(self.layers):
            # if is the last layer
            if index == len(self.layers) - 1:
                layer.update_weights_l(self.all_layers_activations[index], learning_rate, lineal_errors)
            else:
                layer.update_weights(self.all_layers_activations[index], learning_rate, pattern_error)

    def fit(self,
            inputs: list[list[float]],
            outputs: list[list[float]],
            learning_rate: float = 0.3,
            tolerance: float = 0.1,
            epochs: int = 30) -> bool:
        for i in range(epochs):
            iteration_error = 0.0

            for index, inp in enumerate(inputs):
                correct_output = outputs[index]
                self.all_layers_activations = []
                predicted_output = self.eval(inp)

                lineal_errors = [correct - predicted for predicted, correct in
                                 zip(predicted_output, correct_output)]
                pattern_error = sum(map(lambda x: abs(x), lineal_errors)) / float(len(correct_output))
                iteration_error += pattern_error

                self._update_weights(learning_rate, pattern_error, lineal_errors)

            iteration_error /= len(inputs)
            self.errors.append(iteration_error)
            if iteration_error <= tolerance:
                return True

        return False
