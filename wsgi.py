"""Gunicorn entrypoint: `gunicorn wsgi:app`.

The model and the openSMILE extractor are built once at import, so they are
loaded per worker process rather than per request.
"""
from main import create_app

app = create_app()
