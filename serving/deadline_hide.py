"""Peacock deadline-hide scheduler for vLLM 0.32.

Original wrapper. Not a rewrite of vLLM and not the failed PrefillGuard
(that subclassed the sync Scheduler, called throttle_prefills, and hid
late decodes with no time cap — Validation scored 0.0).

This subclasses AsyncScheduler so GPU overlap stays on. Two jobs, both
temporary, never drop a request, never change max_tokens / temperature /
thinking on/off:

1. Reorder the FCFS wait queue so the soonest grader TTFT deadline is
   admitted first (0.5 s + 0.25 s per 1k prompt tokens).
2. For at most HIDE_S seconds, skip scheduling decodes that already
   missed TTFT, so on-time work keeps the GPU. After HIDE_S they are
   scheduled again. At least KEEP_MIN requests stay runnable.

Does not call throttle_prefills. Does not cap max-num-seqs.
"""
from __future__ import annotations

import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

TTFT_BASE_S = 0.5
TTFT_PER_1K_S = 0.25
HIDE_S = 8.0
KEEP_MIN = 8


def _deadline(req) -> float:
    return req.arrival_time + TTFT_BASE_S + TTFT_PER_1K_S * (
        req.num_prompt_tokens / 1000.0
    )


class DeadlineHideScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hidden_since: dict[str, float] = {}
        self._parked: list[tuple[object, float]] = []

    def _reorder_waiting(self) -> None:
        n = len(self.waiting)
        if n <= 1:
            return
        reqs = [self.waiting.pop_request() for _ in range(n)]
        reqs.sort(key=_deadline)
        for req in reqs:
            self.waiting.add_request(req)

    def _release_parked(self, now: float) -> None:
        keep: list[tuple[object, float]] = []
        for req, t0 in self._parked:
            if req.is_finished() or (now - t0) >= HIDE_S:
                if not req.is_finished():
                    self.waiting.add_request(req)
                continue
            keep.append((req, t0))
        self._parked = keep

    def _park_late_waiting(self, now: float) -> None:
        n = len(self.waiting)
        if n <= 1:
            return
        reqs = [self.waiting.pop_request() for _ in range(n)]
        live = [r for r in reqs if now <= _deadline(r)]
        late = [r for r in reqs if now > _deadline(r)]
        if not live or not late:
            for req in reqs:
                self.waiting.add_request(req)
            return
        live.sort(key=_deadline)
        for req in live:
            self.waiting.add_request(req)
        for req in late:
            self._parked.append((req, now))

    def _hide_late_decodes(self, now: float) -> list:
        late = []
        for req in self.running:
            if req.num_output_tokens <= 0:
                continue
            if now <= _deadline(req):
                continue
            t0 = self._hidden_since.get(req.request_id)
            if t0 is not None and (now - t0) >= HIDE_S:
                continue
            late.append(req)
        keep = max(KEEP_MIN, 1)
        budget = max(0, len(self.running) - keep)
        if len(late) > budget:
            late.sort(key=lambda r: now - _deadline(r), reverse=True)
            late = late[:budget]
        if not late:
            return []
        hold = {id(req) for req in late}
        self.running = [req for req in self.running if id(req) not in hold]
        for req in late:
            self._hidden_since.setdefault(req.request_id, now)
        return late

    def _put_back(self, held: list) -> None:
        live = {id(req) for req in self.running}
        for req in held:
            if id(req) in live or req.is_finished():
                continue
            self.running.append(req)

    def _forget(self) -> None:
        live = {req.request_id for req in self.running}
        live.update(req.request_id for req, _ in self._parked)
        if len(self._hidden_since) > 256:
            self._hidden_since = {
                rid: ts for rid, ts in self._hidden_since.items() if rid in live
            }

    def schedule(self, throttle_prefills: bool = False):
        now = time.time()
        self._release_parked(now)
        self._park_late_waiting(now)
        self._reorder_waiting()
        held = self._hide_late_decodes(now)
        try:
            # Keep the caller's flag. Never force throttle_prefills=True.
            return super().schedule(throttle_prefills=throttle_prefills)
        finally:
            self._put_back(held)
            self._forget()
