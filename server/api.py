from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import rbf, perceptron

app = FastAPI()

origins = [
    # Add other origins here
    # You can add "*" for allow all origins
    "*",
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
