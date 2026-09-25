# infer_server.py

Local OpenAI-style chat server with a built-in chat UI (`GET /`).

```bash
python infer_server.py --model ./runs/linear --device auto --dtype auto \
    --host 127.0.0.1 --port 8080 --max-prompt-tokens 512 --api-token SECRET
```

| Flag | Default | What it does |
|---|---|---|
| `--model` / `--device` / `--dtype` | `./runs/linear` / `auto` / `auto` | engine selection |
| `--host` / `--port` | `127.0.0.1` / `8080` | bind address; non-loopback binds warn (and require `--api-token`) |
| `--max-prompt-tokens` | `512` | prompt cap (= model window); max `4096` |
| `--api-token` | none (or `SMAUL_API_TOKEN` env) | require `Authorization: Bearer <token>` (or `X-API-Key`) |

## Endpoints

- `GET /health` -- `{"status": "ok", "device": ...}` (no auth).
- `GET /v1/models` -- static model list.
- `POST /v1/chat/completions` -- `{messages, max_tokens, temperature, top_p, top_k,
  repetition_penalty, stream, system}`. Roles are limited to `user`/`assistant`/`tool`
  (422 otherwise); at most 100 messages, 50k chars each, 10k-char `system`.

## Behavior

- Generation runs in worker threads (`asyncio.to_thread`): the event loop never blocks and
  concurrent requests do not starve each other (no global lock).
- A fast character guard rejects oversized bodies with 413 *before* tokenization; token
  overflows also return 413.
- Disconnected SSE clients stop promptly (`is_disconnected` is polled); mid-stream errors
  arrive as an SSE `finish_reason: "error"` event instead of a silent cutoff.
- There is no rate limiting: front the server or keep it on loopback if you expose it.
