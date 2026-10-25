"""Peacock deadline-prefill scheduler for vLLM 0.32.

Original. Not PrefillGuard (sync + hide, Val 0.0) and not DeadlineHide
(hid running decodes + parked late waits, Val 66.4).

Subclass AsyncScheduler. Never remove a request from running. Never drop.
Never change max_tokens / temperature / thinking on/off.

1. Reorder the FCFS wait queue by grader TTFT deadline
   (0.5 s + 0.25 s per 1k prompt tokens), soonest first.
2. If the last engine step was in (45 ms, 500 ms) — a real busy step,
   not an idle gap — call vLLM's own throttle_prefills so new / in-flight
   long prefills wait one step and decodes keep the GPU.
"""
from __future__ import annotations

import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

TTFT_BASE_S = 0.5
TTFT_PER_1K_S = 0.25
BUSY_MIN_S = 0.045
BUSY_MAX_S = 0.500


def _deadline(req) -> float:
    return req.arrival_time + TTFT_BASE_S + TTFT_PER_1K_S * (
        req.num_prompt_tokens / 1000.0
    )


class DeadlinePrefillScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t_mono = time.monotonic()
        self._last_step_s = 0.0

    def _reorder_waiting(self) -> None:
        n = len(self.waiting)
        if n <= 1:
            return
        reqs = [self.waiting.pop_request() for _ in range(n)]
        reqs.sort(key=_deadline)
        for req in reqs:
            self.waiting.add_request(req)

    def schedule(self, throttle_prefills: bool = False):
        now = time.monotonic()
        self._last_step_s = now - self._t_mono
        self._t_mono = now
        busy = BUSY_MIN_S < self._last_step_s < BUSY_MAX_S
        self._reorder_waiting()
        return super().schedule(throttle_prefills=busy or throttle_prefills)
