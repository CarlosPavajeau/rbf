from fastapi import APIRouter
from models.rbfresponse import InitRbf, FitRbf, EvalRbf, RbfResponse, RadialCenterResponse, NodeResponse
from rbf import Rbf
from networks import Networks

router = APIRouter(prefix='/rbf', tags=['rbf'])
networks = Networks()


def _make_response(rbf: Rbf) -> RbfResponse:
    return RbfResponse(
        num_inputs=rbf.num_inputs,
        radial_centers=[RadialCenterResponse(
            num_inputs=radial.num_inputs,
            centroids=radial.centroids
        ) for _, radial in enumerate(rbf.radial_centers)],
        errors=rbf.errors,
        output_node=NodeResponse(
            num_inputs=rbf.output_node.num_inputs,
            weights=[w for _, w in enumerate(rbf.output_node.weights)],
            bias=rbf.output_node.bias
        )
    )


@router.get('/')
async def get_info():
    return networks.rbf


@router.post('/init')
async def init_rbf(info: InitRbf):
    networks.rbf = Rbf(info.num_inputs, info.radial_centers)
    return _make_response(networks.rbf)


@router.post('/fit')
async def fit_rbf(info: FitRbf):
    networks.rbf.fit(info.inputs, info.outputs, info.tolerance, info.epochs)
    return _make_response(networks.rbf)


@router.patch('/eval')
async def eval_rbf(info: EvalRbf):
    output = networks.rbf.eval(info.inputs)
    return output


@router.post('/set')
async def set_rbf(info: RbfResponse):
    if networks.rbf is None:
        networks.rbf = Rbf(info.num_inputs, len(info.radial_centers))

    networks.rbf.num_inputs = info.num_inputs
    networks.rbf.radial_centers = info.radial_centers
    networks.rbf.output_node = info.output_node

    return _make_response(networks.rbf)
