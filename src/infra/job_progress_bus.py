from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Protocol

_SENTINEL: object = object()


class TrackedJob(Protocol):
    """Structural contract for anything published on the bus - currently
    TrainingJob and DubbingJob. Both are simple mutable progress records
    identified by job_id, so one generic bus serves either.
    """

    job_id: str


@dataclass(frozen=True)
class ProgressEvent:
    job_id: str
    job_snapshot: TrackedJob


class JobProgressBus:
    """In-process pub/sub channel carrying engine progress to the UI.

    Engines push updates via publish(); the webui layer subscribes per job
    (e.g. to stream Server-Sent Events) without ever importing an engine
    module directly.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Queue]] = defaultdict(list)
        self._latest_by_job_id: dict[str, TrackedJob] = {}

    def subscribe(self, job_id: str) -> Queue:
        queue: Queue = Queue()
        self._subscribers[job_id].append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: Queue) -> None:
        if queue in self._subscribers.get(job_id, []):
            self._subscribers[job_id].remove(queue)

    def publish(self, job: TrackedJob) -> None:
        self._latest_by_job_id[job.job_id] = job
        for queue in self._subscribers.get(job.job_id, []):
            queue.put(ProgressEvent(job_id=job.job_id, job_snapshot=job))

    def latest_for_job(self, job_id: str) -> TrackedJob | None:
        """Most recent snapshot for one run (training or dubbing). job_id
        is known to the caller as soon as the run starts, not just once it
        completes, so callers can poll immediately.
        """
        return self._latest_by_job_id.get(job_id)

    def close(self, job_id: str) -> None:
        for queue in self._subscribers.get(job_id, []):
            queue.put(_SENTINEL)
        self._subscribers.pop(job_id, None)

    @staticmethod
    def is_closed_event(item: object) -> bool:
        return item is _SENTINEL

    @staticmethod
    def poll(queue: Queue, timeout: float = 1.0) -> ProgressEvent | None:
        try:
            item = queue.get(timeout=timeout)
        except Empty:
            return None
        if JobProgressBus.is_closed_event(item):
            return None
        assert isinstance(item, ProgressEvent)
        return item


progress_bus = JobProgressBus()
