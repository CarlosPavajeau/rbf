from fastapi import APIRouter
from models.perceptron import InitPerceptron, FitPerceptron, EvalPerceptron, Perceptron

router = APIRouter(prefix='/perceptron', tags=['perceptron'])


@router.get('/')
async def get_info():
    return 'hello'


@router.post('/init')
async def init_perceptron(info: InitPerceptron):
    return info


@router.post('/fit')
async def fit_perceptron(info: FitPerceptron):
    return info


@router.get('/eval')
async def eval_perceptron(info: EvalPerceptron):
    return info


@router.post('/set')
async def set_perceptron(info: Perceptron):
    return info
