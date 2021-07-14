#!/bin/bash

PORT=80

uvicorn app.app:app --host 0.0.0.0 --port "${PORT}"
