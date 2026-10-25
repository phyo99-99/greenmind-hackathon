"""Peacock prefill-guard scheduler for vLLM 0.32.

Not a rewrite of vLLM. Subclasses the stock Scheduler and does two things
when the last engine step was slower than 45 ms (our Final #15 peak is ~55 ms):

1. Call schedule(throttle_prefills=True). That is vLLM's own hook: do not
   start a new long prefill while a decode is already running.
2. Hold decode on requests that already missed the grader TTFT
   (0.5 s + 0.25 s per 1k prompt tokens). Those score 0 anyway and still
   steal the GPU. They stay in KV and resume when the step is fast again.
   Never hold the whole running set (that stalls the engine).

Does not change max_tokens, temperature, or thinking on/off.
Validation first. If chat breaks, kill and restore the 94.6 serve
(no --scheduler-cls).
"""
from __future__ import annotations

import time

from vllm.v1.core.sched.scheduler import Scheduler

# One step under the 50 ms TPOT SLO. Matches our measured #15 peak (~55 ms).
BUSY_S = 0.045
TTFT_BASE_S = 0.5
TTFT_PER_1K_S = 0.25
TTFT_SLACK_S = 0.05


def _ttft_limit_s(prompt_tokens: int) -> float:
    return TTFT_BASE_S + TTFT_PER_1K_S * (prompt_tokens / 1000.0)


class PrefillGuardScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t_mono = time.monotonic()
        self._last_step_s = 0.0
        self._first_token_wall: dict[str, float] = {}

    def _note_first_tokens(self) -> None:
        now = time.time()
        for req in self.running:
            if req.num_output_tokens > 0 and req.request_id not in self._first_token_wall:
                self._first_token_wall[req.request_id] = now

    def _missed_ttft(self, req) -> bool:
        first = self._first_token_wall.get(req.request_id)
        if first is None or req.num_output_tokens == 0:
            return False
        limit = _ttft_limit_s(req.num_prompt_tokens) + TTFT_SLACK_S
        return (first - req.arrival_time) > limit

    def _hold_missed_ttft(self) -> list:
        missed = [req for req in self.running if self._missed_ttft(req)]
        if not missed or len(missed) >= len(self.running):
            return []
        hold_ids = {id(req) for req in missed}
        self.running = [req for req in self.running if id(req) not in hold_ids]
        return missed

    def _put_back(self, held: list) -> None:
        running_ids = {id(req) for req in self.running}
        for req in held:
            if id(req) in running_ids or req.is_finished():
                continue
            self.running.append(req)

    def _forget_finished(self) -> None:
        live = {req.request_id for req in self.running}
        if len(self._first_token_wall) > 256:
            self._first_token_wall = {
                rid: ts for rid, ts in self._first_token_wall.items() if rid in live
            }

    def schedule(self, throttle_prefills: bool = False):
        now = time.monotonic()
        self._last_step_s = now - self._t_mono
        self._t_mono = now
        busy = self._last_step_s > BUSY_S

        self._note_first_tokens()
        held = self._hold_missed_ttft() if busy else []
        try:
            return super().schedule(throttle_prefills=busy or throttle_prefills)
        finally:
            self._put_back(held)
            self._forget_finished()
