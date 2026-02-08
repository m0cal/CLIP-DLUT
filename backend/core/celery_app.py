from celery import Celery
from .config import settings

celery_app = Celery(
    "clip_dlut_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["backend.tasks.processing"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    # Limit concurrency to 1 if on GPU to avoid OOM
    worker_concurrency=1,
    worker_prefetch_multiplier=1
)
