# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI

import tensorrt_llm.serve.openai_server as openai_server_module
from tensorrt_llm.serve.openai_server import (
    OpenAIServer,
    _multimodal_processor_torch_initializer,
    _MultimodalRequestBodyLimitMiddleware,
)

pytestmark = pytest.mark.cpu_only


class _FakeMultimodalInputProcessor:
    pass


def test_limits_multimodal_processor_torch_threads(monkeypatch):
    monkeypatch.setattr(
        openai_server_module, "BaseMultimodalInputProcessor", _FakeMultimodalInputProcessor
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: 72)
    observed = []
    monkeypatch.setattr(torch, "set_num_threads", observed.append)

    initializer = _multimodal_processor_torch_initializer(
        SimpleNamespace(args=SimpleNamespace(gather_generation_logits=False)),
        _FakeMultimodalInputProcessor(),
    )
    assert initializer is not None
    initializer()

    assert observed == [16]


@pytest.mark.parametrize(
    "single_process,gather_generation_logits",
    [(True, False), (False, True)],
)
def test_does_not_limit_threads_shared_with_gpu_worker(
    monkeypatch, single_process, gather_generation_logits
):
    monkeypatch.setattr(
        openai_server_module, "BaseMultimodalInputProcessor", _FakeMultimodalInputProcessor
    )
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1" if single_process else "0")
    observed = []
    monkeypatch.setattr(torch, "set_num_threads", observed.append)

    initializer = _multimodal_processor_torch_initializer(
        SimpleNamespace(args=SimpleNamespace(gather_generation_logits=gather_generation_logits)),
        _FakeMultimodalInputProcessor(),
    )

    assert initializer is None
    assert observed == []


async def _run_body_limit(
    chunks: list[bytes],
    *,
    max_bytes: int,
    path: str = "/v1/chat/completions",
    content_length: int | None = None,
):
    received_body = bytearray()
    observed_state = {}

    async def app(scope, receive, send):
        while True:
            message = await receive()
            received_body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        observed_state.update(scope.get("state", {}))
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index + 1 < len(chunks),
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive():
        return messages.pop(0)

    sent = []

    async def send(message):
        sent.append(message)

    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
    }
    middleware = _MultimodalRequestBodyLimitMiddleware(app, max_bytes)
    await middleware(scope, receive, send)
    return sent[0]["status"], bytes(received_body), observed_state


@pytest.mark.asyncio
async def test_content_length_rejects_before_reading_body():
    status, body, _ = await _run_body_limit([b"not read"], max_bytes=8, content_length=9)

    assert status == 413
    assert body == b""


@pytest.mark.asyncio
async def test_chunked_body_is_rejected_at_limit():
    status, body, _ = await _run_body_limit([b"12345", b"678901"], max_bytes=10)

    assert status == 413
    assert body == b"12345"


@pytest.mark.asyncio
async def test_chunked_body_stays_413_through_fastapi():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def route(payload: dict):
        return payload

    messages = [
        {"type": "http.request", "body": b'{"value":', "more_body": True},
        {"type": "http.request", "body": b'"too long"}', "more_body": False},
    ]

    async def receive():
        return messages.pop(0)

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1),
        "server": ("test", 80),
        "state": {},
    }
    middleware = _MultimodalRequestBodyLimitMiddleware(app, max_bytes=10)
    await middleware(scope, receive, send)

    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_accepted_body_bytes_are_recorded_for_decoded_limit():
    status, body, state = await _run_body_limit([b"12345", b"67890"], max_bytes=10)

    assert status == 204
    assert body == b"1234567890"
    assert state["multimodal_request_body_bytes"] == 10


@pytest.mark.asyncio
async def test_non_multimodal_route_is_unchanged():
    status, body, _ = await _run_body_limit([b"123456789"], max_bytes=8, path="/v1/completions")

    assert status == 204
    assert body == b"123456789"


@pytest.mark.asyncio
async def test_total_limit_serializes_multimodal_preprocessing():
    server = object.__new__(OpenAIServer)
    server._mm_cpu_request_slots = asyncio.BoundedSemaphore(1)

    await server._acquire_mm_cpu_slot()
    next_request = asyncio.create_task(server._acquire_mm_cpu_slot())
    await asyncio.sleep(0)
    assert not next_request.done()

    server._mm_cpu_request_slots.release()
    await next_request
    server._mm_cpu_request_slots.release()
