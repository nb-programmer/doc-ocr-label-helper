from fastapi import FastAPI


def init_routes(app: FastAPI):
    from . import dataset

    dataset.init_app(app)
