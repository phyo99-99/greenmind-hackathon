"""vLLM middleware: set thinking_token_budget only.

Does not change max_tokens, temperature, or thinking on/off.
"""
import json

PATHS = ("/v1/chat/completions", "/v1/completions")


def think_budget(data):
    if "thinking_token_budget" in data:
        return None
    if (data.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
        return None
    mt = data.get("max_tokens", data.get("max_completion_tokens"))
    if not isinstance(mt, int) or mt <= 0:
        return None
    reserve = min(1024, max(24, mt // 2))
    return max(mt - reserve, 0)


class ThinkingBudgetMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in PATHS
        ):
            await self.app(scope, receive, send)
            return

        chunks = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        body = b"".join(chunks)

        try:
            data = json.loads(body)
        except ValueError:
            data = None

        if isinstance(data, dict):
            budget = think_budget(data)
            if budget is not None:
                data["thinking_token_budget"] = budget
                body = json.dumps(data, ensure_ascii=False).encode()

        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        await self.app(dict(scope, headers=headers), replay, send)
