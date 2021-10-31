from fastapi import APIRouter
from models.perceptronresponse import InitPerceptron, FitPerceptron, EvalPerceptron, PerceptronResponse, LayerResponse, \
    NodeResponse
from perceptron import Perceptron
from networks import Networks

router = APIRouter(prefix='/perceptron', tags=['perceptron'])
networks = Networks()


def _make_response(perceptron: Perceptron) -> PerceptronResponse:
    return PerceptronResponse(
        num_inputs=perceptron.num_inputs,
        num_layers=perceptron.num_layers,
        layers=[LayerResponse(
            num_inputs=layer.num_inputs,
            activation_function_name=layer.activation_function_name,
            nodes=[NodeResponse(
                num_inputs=node.num_inputs,
                weights=[w for _, w in enumerate(node.weights)],
                bias=node.bias
            ) for _, node in enumerate(layer.nodes)]
        ) for _, layer in enumerate(perceptron.layers)],
        errors=perceptron.errors
    )


@router.get('/')
async def get_info():
    return _make_response(networks.perceptron)


@router.post('/init')
async def init_perceptron(info: InitPerceptron):
    networks.perceptron = Perceptron(info.num_inputs, info.num_layers, info.nodes_per_layer,
                                     info.activation_functions_names)
    return _make_response(networks.perceptron)


@router.post('/fit')
async def fit_perceptron(info: FitPerceptron):
    networks.perceptron.fit(info.inputs, info.outputs, info.learning_rate, info.tolerance, info.epochs)
    return _make_response(networks.perceptron)


@router.patch('/eval')
async def eval_perceptron(info: EvalPerceptron):
    output = networks.perceptron.eval(info.inputs)
    return output


def _make_perceptron(info: PerceptronResponse):
    nodes_per_layer = [len(layer.nodes) for _, layer in enumerate(info.layers)]
    activation_functions_names = [layer.activation_function_name for _, layer in enumerate(info.layers)]

    networks.perceptron = Perceptron(info.num_inputs, info.num_layers, nodes_per_layer, activation_functions_names)

    for index, layer in enumerate(networks.perceptron.layers):
        for idx, node in enumerate(layer.nodes):
            node.weights = info.layers[index].nodes[idx].weights
            node.bias = info.layers[index].nodes[idx].bias


@router.post('/set')
async def set_perceptron(info: PerceptronResponse):
    _make_perceptron(info)
    return _make_response(networks.perceptron)
