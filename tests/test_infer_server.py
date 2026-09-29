import json
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from infer_server import create_app


class Engine:
    """A working double.

    The previous one returned "prompt exceeds limit" from chat_prompt, so every
    single request it was given stopped at the 413 guard and no success path --
    streaming or not -- was ever exercised.
    """

    device = "cpu"

    class Model:
        def num_parameters(self):
            return 100

    model = Model()

    def __init__(self, pieces=("Hello", " ", "world"), encode_error=None,
                 prompt_error=None, stream_error=None):
        self.release = threading.Event()
        self.pieces = list(pieces)
        self.encode_error = encode_error
        self.prompt_error = prompt_error
        self.stream_error = stream_error
        self.calls = []
        self.streamed = 0
        self.counted = []

    def encode(self, text):
        if self.encode_error:
            raise self.encode_error
        return text.split()

    def chat_prompt(self, messages, system=None):
        if self.prompt_error:
            raise self.prompt_error
        return "PROMPT:" + " ".join(m["content"] for m in messages)

    def stream(self, prompt, **kwargs):
        self.calls.append(("stream", prompt, kwargs))
        for i, piece in enumerate(self.pieces):
            if self.stream_error is not None and i == 1:
                raise self.stream_error
            self.streamed += 1
            self.counted.append(piece)
            yield piece

    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_k=50,
                 top_p=0.95, repetition_penalty=1.05):
        self.calls.append(("generate", prompt, (max_new_tokens, temperature,
                                                 top_k, top_p, repetition_penalty)))
        return "".join(self.pieces)


def _client(engine=None, **kw):
    return TestClient(create_app(engine or Engine(), **kw))


def _post(client, **over):
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    body.update(over)
    return client.post("/v1/chat/completions", json=body)


# --- the success paths, which had no coverage at all -----------------------

def test_non_streaming_completion_returns_openai_shape():
    eng = Engine()
    response = _post(_client(eng))
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert isinstance(data["created"], int)
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"] == {"role": "assistant", "content": "Hello world"}
    assert choice["finish_reason"] == "stop"


def test_non_streaming_passes_the_sampling_arguments_through():
    eng = Engine()
    _post(_client(eng), max_tokens=13, temperature=0.1, top_k=3, top_p=0.5,
          repetition_penalty=1.2)
    kind, prompt, args = eng.calls[0]
    assert kind == "generate"
    assert prompt == "PROMPT:hi"
    assert args == (13, 0.1, 3, 0.5, 1.2), "sampling args were reordered or dropped"


def test_streaming_emits_sse_deltas_then_stop_then_done():
    response = _post(_client(), stream=True)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    frames = [f for f in response.text.split("\n\n") if f.startswith("data: ")]
    assert frames[-1] == "data: [DONE]"
    payloads = [json.loads(f[len("data: "):]) for f in frames[:-1]]
    assert "".join(p["choices"][0]["delta"].get("content", "")
                   for p in payloads) == "Hello world"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(p["object"] == "chat.completion.chunk" for p in payloads)
    assert all(p["id"].startswith("chatcmpl-") for p in payloads)


def test_streaming_and_non_streaming_agree_on_the_text():
    streamed = _post(_client(), stream=True).text
    done = _post(_client(), stream=False).json()
    joined = "".join(json.loads(f[len("data: "):])["choices"][0]["delta"].get("content", "")
                    for f in streamed.split("\n\n")
                    if f.startswith("data: ") and f != "data: [DONE]")
    assert joined == done["choices"][0]["message"]["content"]


def test_system_prompt_is_accepted():
    eng = Engine()
    assert _post(_client(eng), system="Be terse.").status_code == 200
    assert eng.calls[0][1] == "PROMPT:hi"      # the double folds system in itself


def test_multiple_messages_are_all_forwarded():
    eng = Engine()
    _post(_client(eng), messages=[{"role": "user", "content": "one"},
                                  {"role": "assistant", "content": "two"},
                                  {"role": "user", "content": "three"}])
    assert eng.calls[0][1] == "PROMPT:one two three"


def test_tool_role_is_accepted():
    assert _post(_client(), messages=[{"role": "tool", "content": "result"}]).status_code == 200


def test_a_system_role_inside_messages_is_rejected():
    """A documented restriction, not an oversight: system has its own field.

    OpenAI clients habitually send the system turn as messages[0] with role
    "system". This schema accepts only user/assistant/tool and puts the system
    prompt in a separate top-level `system` field, so that habit gets a 422 with
    a clear enum error rather than being quietly reinterpreted. Pinned so the
    restriction is a decision on record.
    """
    response = _post(_client(), messages=[{"role": "system", "content": "rules"},
                                         {"role": "user", "content": "q"}])
    assert response.status_code == 422
    assert "system" in response.text


def test_the_top_level_system_field_is_the_supported_way():
    eng = Engine()
    assert _post(_client(eng), system="rules").status_code == 200
    assert eng.calls, "the request never reached generation"


# --- static endpoints -------------------------------------------------------

def test_index_serves_the_chat_ui():
    response = _client().get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>SmaulLinear</title>" in response.text


def test_models_lists_the_one_model():
    data = _client().get("/v1/models").json()
    assert data["object"] == "list"
    assert data["data"] == [{"id": "smaul-linear", "object": "model",
                             "owned_by": "SmaulNative"}]


def test_health_reports_the_device():
    data = _client().get("/health").json()
    assert data == {"status": "ok", "device": "cpu"}


# --- auth -------------------------------------------------------------------

def test_no_token_configured_means_open(monkeypatch):
    monkeypatch.delenv("SMAUL_API_TOKEN", raising=False)
    assert _post(_client()).status_code == 200


def test_api_token_is_required_when_configured(monkeypatch):
    monkeypatch.delenv("SMAUL_API_TOKEN", raising=False)
    client = _client(api_token="s3cret")
    assert _post(client).status_code == 401
    assert _post(client, ).json()["detail"] == "invalid API token"


def _auth_post(client, header, value, **over):
    """Headers must be supplied at send time; TestClient sends immediately."""
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    body.update(over)
    return client.post("/v1/chat/completions", json=body, headers={header: value})


def test_bearer_and_x_api_key_both_authenticate(monkeypatch):
    monkeypatch.delenv("SMAUL_API_TOKEN", raising=False)
    client = _client(api_token="s3cret")
    assert _auth_post(client, "Authorization", "Bearer s3cret").status_code == 200
    assert _auth_post(client, "X-API-Key", "s3cret").status_code == 200, \
        "X-API-Key is an accepted alternative to Bearer"


def test_a_wrong_token_is_rejected(monkeypatch):
    monkeypatch.delenv("SMAUL_API_TOKEN", raising=False)
    client = _client(api_token="s3cret")
    # A prefix of the real token must not pass: the comparison is exact and
    # constant-time, not a startswith.
    assert _auth_post(client, "Authorization", "Bearer s3cretX").status_code == 401
    assert _auth_post(client, "Authorization", "Bearer ").status_code == 401
    assert _auth_post(client, "X-API-Key", "s3cre").status_code == 401
    assert _auth_post(client, "X-API-Key", "s3cretX").status_code == 401


def test_env_var_supplies_the_token(monkeypatch):
    monkeypatch.setenv("SMAUL_API_TOKEN", "from-env")
    client = _client()
    assert _post(client).status_code == 401
    assert _auth_post(client, "Authorization", "Bearer from-env").status_code == 200


def test_the_token_guards_only_the_generation_endpoint(monkeypatch):
    """Health and the model list are unauthenticated, as documented."""
    monkeypatch.delenv("SMAUL_API_TOKEN", raising=False)
    client = _client(api_token="s3cret")
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert client.get("/").status_code == 200
    assert _post(client).status_code == 401


# --- request validation -----------------------------------------------------

@pytest.mark.parametrize("over,code", [
    (dict(max_tokens=0), 422), (dict(max_tokens=65_537), 422),
    (dict(temperature=-0.1), 422), (dict(temperature=5.1), 422),
    (dict(top_p=0.0), 422), (dict(top_p=1.1), 422),
    (dict(top_k=-1), 422),
    (dict(repetition_penalty=0.4), 422), (dict(repetition_penalty=2.1), 422),
    (dict(messages=[]), 422),
    (dict(messages=[{"role": "nope", "content": "x"}]), 422),
    (dict(messages=[{"role": "user", "content": "x" * 50_001}]), 422),
])
def test_out_of_range_fields_are_rejected_by_the_schema(over, code):
    assert _post(_client(), **over).status_code == code


def test_missing_messages_is_rejected():
    assert _client().post("/v1/chat/completions", json={}).status_code == 422


def test_too_many_messages_are_rejected():
    body = {"messages": [{"role": "user", "content": "x"}] * 101, "stream": False}
    assert _client().post("/v1/chat/completions", json=body).status_code == 422


def test_max_prompt_tokens_must_be_sane():
    with pytest.raises(ValueError, match="positive"):
        create_app(Engine(), max_prompt_tokens=0)
    with pytest.raises(ValueError, match="262144"):
        create_app(Engine(), max_prompt_tokens=262_145)


# --- the two size guards, which are different checks -----------------------

def test_a_char_overload_is_rejected_before_tokenizing():
    """The cheap guard must fire first: it exists to avoid the expensive call."""
    eng = Engine()
    # char budget is max_prompt_tokens*4 + 10_000 = 10_032 here. The content must
    # stay under the schema's own 50_000 cap or the 422 fires first, which is a
    # different and earlier check.
    response = _post(_client(eng, max_prompt_tokens=8),
                     messages=[{"role": "user", "content": "x " * 20_000}])
    assert response.status_code == 413
    assert "chars" in response.json()["detail"]
    assert not eng.calls, "generation ran despite the guard"


def test_a_token_overload_is_rejected_after_tokenizing():
    eng = Engine()
    # PROMPT:one two three is three tokens under the double's whitespace encode.
    response = _post(_client(eng, max_prompt_tokens=1),
                     messages=[{"role": "user", "content": "one two three"}])
    assert response.status_code == 413
    detail = response.json()["detail"]
    # Reports the limit and the actual count, not the prompt text.
    assert "prompt exceeds 1 tokens (3)" == detail


# --- engine failures surface as 4xx, not tracebacks -------------------------

def test_a_bad_prompt_is_a_400():
    eng = Engine(prompt_error=ValueError("role not allowed"))
    response = _post(_client(eng))
    assert response.status_code == 400
    assert "role not allowed" in response.json()["detail"]


def test_a_tokenizer_failure_is_a_400():
    eng = Engine(encode_error=RuntimeError("vocab mismatch"))
    response = _post(_client(eng))
    assert response.status_code == 400
    assert "could not tokenize" in response.json()["detail"]


def test_a_bad_generation_argument_is_a_400():
    class Strict(Engine):
        def generate(self, *a, **k):
            raise ValueError("temperature must be > 0")

    response = _post(_client(Strict()))
    assert response.status_code == 400
    assert "temperature" in response.json()["detail"]


# --- a failure mid-stream is surfaced, not silently truncated ---------------

def test_a_mid_stream_error_arrives_as_an_error_chunk():
    """Truncating the stream without saying so is the worst outcome here."""
    eng = Engine(pieces=("a", "b", "c"), stream_error=RuntimeError("kernel died"))
    text = _post(_client(eng), stream=True).text
    frames = [f for f in text.split("\n\n") if f.startswith("data: ")]
    payloads = [json.loads(f[len("data: "):]) for f in frames if f != "data: [DONE]"]
    errors = [p for p in payloads if p["choices"][0]["finish_reason"] == "error"]
    assert errors, f"no error chunk in {payloads}"
    assert "kernel died" in json.dumps(errors[0])


def test_a_mid_stream_error_still_terminates_the_stream():
    eng = Engine(stream_error=RuntimeError("boom"))
    text = _post(_client(eng), stream=True).text
    assert text.rstrip().endswith("data: [DONE]"), "the stream never terminated"


# --- the producer thread ----------------------------------------------------
#
# Driven by calling the route handler directly rather than through TestClient.
# A sync TestClient close on a half-read SSE response can block until the
# producer finishes, which is what wedged the suite when this was first written
# as an HTTP test. Calling the endpoint with a stub Request gives the same
# control over is_disconnected() and cannot hang.

def _chat_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", None) == "/v1/chat/completions":
            return route.endpoint
    raise AssertionError("chat endpoint not found")


class StubRequest:
    def __init__(self, disconnect_after=0):
        self.checks = 0
        self.disconnect_after = disconnect_after

    async def is_disconnected(self):
        self.checks += 1
        return self.checks > self.disconnect_after


class _CountedEngine(Engine):
    """An endless producer that records how many tokens it actually produced."""

    def __init__(self, interval=0.002):
        super().__init__()
        self.counted = []
        self.interval = interval
        self.running = threading.Event()
        self.release = threading.Event()

    def stream(self, prompt, **kwargs):
        self.calls.append(("stream", prompt, kwargs))
        self.running.set()
        while not self.release.is_set():
            self.counted.append(1)
            time.sleep(self.interval)
            yield "tok "


def _settle_observation(engine, **drive_kw):
    """Frames plus before/after production counts, measured inside the loop."""
    import asyncio

    from infer_server import ChatRequest

    app = create_app(engine)
    chat = _chat_endpoint(app)
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
    request = StubRequest(drive_kw.get("disconnect_after", 0))
    max_chunks = drive_kw.get("max_chunks")

    async def go():
        response = await chat(req, request, _auth=None)
        out = []
        stream = response.body_iterator
        try:
            async for frame in stream:
                out.append(frame)
                if max_chunks is not None and len(out) >= max_chunks:
                    break
        finally:
            # What Starlette does when a client disconnects mid-stream, and the
            # only thing that raises GeneratorExit in the generator. Breaking out
            # of an `async for` on its own leaves the generator suspended and open,
            # so without this the measurement watches a stream nobody closed.
            await stream.aclose()
        await asyncio.sleep(0.05)
        at_exit = len(engine.counted)
        await asyncio.sleep(0.4)
        return out, at_exit, len(engine.counted)

    try:
        return asyncio.run(go())
    finally:
        # Only the endless producer needs releasing; a finite one is done.
        engine.release.set()


def test_a_disconnect_stops_the_producer_thread():
    """A client that hangs up must not leave the server generating for it.

    The producer is a daemon thread feeding an asyncio queue, so it has no other
    way to learn the consumer is gone. It used to run to completion regardless,
    holding a thread and every chunk it produced -- up to max_tokens worth -- for
    a response nobody would read.
    """
    eng = _CountedEngine()
    _frames, at_exit, after = _settle_observation(eng, disconnect_after=0)
    assert eng.running.is_set(), "the producer never started"
    # Without this, "the count stopped climbing" is trivially true.
    assert at_exit > 0, "the producer produced nothing, so this proves nothing"
    assert after <= at_exit + 2, (
        f"producer kept generating after the client hung up: "
        f"{at_exit} -> {after} tokens")


def test_closing_the_response_early_stops_the_producer():
    """The same leak, reached by closing the generator rather than disconnecting.

    A client that stops reading a streamed response closes the generator, which
    arrives as GeneratorExit. The stop signal has to sit in a finally, or this
    path leaks exactly as the explicit-disconnect one did.
    """
    eng = _CountedEngine()
    _frames, at_exit, after = _settle_observation(eng, disconnect_after=10_000,
                                                  max_chunks=2)
    assert eng.running.is_set(), "the producer never started"
    assert at_exit > 0, "the producer produced nothing, so this proves nothing"
    assert after <= at_exit + 2, (
        f"producer kept generating after the response was closed: "
        f"{at_exit} -> {after} tokens")


def test_a_normal_finish_still_emits_stop_and_done():
    """The stop signal must not truncate a stream that ran to completion."""
    frames, _at, _after = _settle_observation(
        Engine(pieces=("a", "b")), disconnect_after=10_000)
    text = "".join(frames)
    assert text.rstrip().endswith("data: [DONE]")
    # startswith, not ==: every frame carries its trailing "\n\n".
    payloads = [json.loads(f[len("data: "):]) for f in frames
                if f.startswith("data: ") and not f.startswith("data: [DONE]")]
    assert "".join(p["choices"][0]["delta"].get("content", "")
                   for p in payloads) == "ab"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
