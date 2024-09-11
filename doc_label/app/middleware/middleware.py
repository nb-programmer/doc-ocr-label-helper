import logging

from fastapi import FastAPI

LOG = logging.getLogger(__name__)


def init_middleware(app: FastAPI):
    from . import cors

    LOG.info("Adding `CORS` middleware")
    cors.init_middleware(app)
