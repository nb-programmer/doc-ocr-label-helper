@REM Launch development server

@echo off

python3 -m uvicorn --factory --reload --host=0.0.0.0 --log-config=config/debug-logger.yml --timeout-graceful-shutdown=10 doc_label:init_app
