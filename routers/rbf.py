from fastapi import APIRouter
from models.rbf import InitRbf, FitRbf, EvalRbf, Rbf

router = APIRouter(prefix='/rbf', tags=['rbf'])


@router.get('/')
async def get_info():
    return "hello"


@router.post('/init')
async def init_rbf(info: InitRbf):
    return info


@router.post('/fit')
async def fit_rbf(info: FitRbf):
    return info


@router.get('/eval')
async def eval_rbf(info: EvalRbf):
    return info


@router.post('/set')
async def set_rbf(info: Rbf):
    return info
