#!/bin/bash
# Wait for tesseract to be ready if needed (optional)
uvicorn app:app --host 0.0.0.0 --port $PORT
