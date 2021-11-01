from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rbf import Rbf
from perceptron import Perceptron
from routers import rbf, perceptron

app = FastAPI()

origins = [
    # Add other origins here
    # You can add "*" for allow all origins
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rbf.router)
app.include_router(perceptron.router)


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


@app.post("/items")
def read_root(item: Item):
    return item.name


if __name__ == '__main__':
    inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    outputs_p = [[1], [0], [0], [0]]
    outputs_r = [1, 0, 0, 0]

    perceptron = Perceptron(2, 2, [2, 1], ['tanh', 'sigmoid', 'linear'])
    perceptron.fit(inputs, outputs_p, epochs=3000, learning_rate=0.1)

    rbf = Rbf(len(inputs[0]), 3)
    train_success = rbf.fit(inputs, outputs_r, tolerance=0.05)

    print(f'Output -> {perceptron.eval([1, 1])}')
