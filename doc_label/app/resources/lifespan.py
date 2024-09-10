from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    from . import dataset

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(dataset.lifespan(app))
        yield
