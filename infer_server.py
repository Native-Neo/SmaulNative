#!/usr/bin/env python3
"""Local SmaulLinear server with chat UI."""

import argparse
import asyncio
import hmac
import json
import os
import time
import uuid
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import uvicorn

from inference import MODEL_WINDOW, LinearInference

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmaulLinear</title><style>
body{margin:0;background:#101010;color:#f1f1f1;font:15px system-ui,sans-serif;height:100vh;overflow:hidden}.app{display:grid;grid-template-columns:260px 1fr;height:100vh}.side{border-right:1px solid #2d2d2d;background:#141414;padding:14px;display:flex;flex-direction:column;gap:12px}.brand{font-weight:700;font-size:18px;padding:8px}.new{border:1px solid #2d2d2d;background:#202020;color:#f1f1f1;border-radius:10px;padding:10px;cursor:pointer}.history{overflow:auto;display:flex;flex-direction:column;gap:4px}.main{min-width:0;display:flex;flex-direction:column}.top{height:58px;border-bottom:1px solid #2d2d2d;display:flex;align-items:center;justify-content:space-between;padding:0 20px}.chat{flex:1;overflow:auto}.messages{max-width:850px;margin:auto;padding:35px 22px 140px}.msg{margin:25px 0}.role{font-size:12px;color:#9b9b9b;margin-bottom:7px}.bubble{line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere}.composer{position:fixed;bottom:0;left:260px;right:0;padding:18px 20px 22px}.box{max-width:850px;margin:auto;background:#171717;border:1px solid #2d2d2d;border-radius:14px;padding:12px}
</style></head><body><div class="app"><aside class="side"><div class="brand">SmaulLinear</div><button class="new" onclick="newChat()">New chat</button><div id="history" class="history"></div></aside><main class="main"><header class="top"><div>Linear <span id="modelmeta"></span></div><div id="status">Ready</div></header><section class="chat" id="chat"><div id="messages" class="messages"></div></section><div class="composer"><div class="box"><textarea id="input" placeholder="Message SmaulLinear…" rows="1" style="width:100%"></textarea><button id="send" onclick="send()">Send</button></div></div></main></div><script>
let messages=[],busy=false;const $=id=>document.getElementById(id),input=$('input');function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function draw(){let m=$('messages');m.innerHTML=messages.map(x=>`<div class="msg"><div class="role">${esc(x.role)}</div><div class="bubble">${esc(x.content)}</div></div>`).join('');$('chat').scrollTop=$('chat').scrollHeight}async function send(){if(busy)return;let text=input.value.trim();if(!text)return;input.value='';messages.push({role:'user',content:text});messages.push({role:'assistant',content:''});draw();busy=true;try{let r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'smaul-linear',messages:messages.slice(0,-1),stream:true,max_tokens:256})});let rd=r.body.getReader(),dec=new TextDecoder(),buf='';while(true){let {done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let parts=buf.split('\n\n');buf=parts.pop();for(let p of parts){if(p.startsWith('data: ')){let d=p.slice(6);if(d==='[DONE]')continue;try{messages[messages.length-1].content+=JSON.parse(d).choices[0].delta.content||'';draw()}catch(e){}}}}catch(e){messages[messages.length-1].content+='[error]'}busy=false;draw()}function newChat(){messages=[];draw()}
</script></body></html>'''


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str = Field(max_length=50_000)


class ChatRequest(BaseModel):
    model: str = "smaul-linear"
    messages: List[Message] = Field(min_length=1, max_length=100)
    max_tokens: int = Field(256, ge=1, le=65536)
    temperature: float = Field(0.7, ge=0, le=5)
    top_p: float = Field(0.95, gt=0, le=1)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.05, ge=0.5, le=2)
    stream: bool = True
    system: Optional[str] = Field(default=None, max_length=10_000)


def create_app(engine: LinearInference, max_prompt_tokens: int = MODEL_WINDOW,
               api_token: Optional[str] = None):
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")
    if max_prompt_tokens > 262144:
        raise ValueError("max_prompt_tokens must be <= 262144")
    app = FastAPI(title="SmaulLinear", version="0.2.0")
    bearer = HTTPBearer(auto_error=False)

    async def check_auth(request: Request,
                         creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
        expected = api_token or os.environ.get("SMAUL_API_TOKEN")
        if not expected:
            return
        token = creds.credentials if creds else request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(str(token), str(expected)):
            raise HTTPException(401, "invalid API token")
        return

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML

    @app.get("/health")
    async def health():
        return {"status": "ok", "device": str(engine.device)}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "smaul-linear", "object": "model", "owned_by": "SmaulNative"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request, _auth=Depends(check_auth)):
        msgs = [m.model_dump() for m in req.messages]
        system = req.system or "You are SmaulLinear, a helpful local AI assistant. Be concise, accurate, and practical."
        # Fast char guard BEFORE expensive tokenization (OOM/CPU guard).
        total_chars = sum(len(m.get("content", "")) for m in msgs) + len(system)
        if total_chars > max_prompt_tokens * 4 + 10_000:
            raise HTTPException(413, f"prompt too large ({total_chars} chars)")
        try:
            prompt = engine.chat_prompt(msgs, system)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            prompt_tokens = len(engine.encode(prompt))
        except Exception as exc:
            raise HTTPException(400, f"could not tokenize prompt: {exc}") from exc
        if prompt_tokens > max_prompt_tokens:
            raise HTTPException(413, f"prompt exceeds {max_prompt_tokens} tokens ({prompt_tokens})")
        created = int(time.time())
        request_id = "chatcmpl-" + uuid.uuid4().hex

        async def chunks():
            try:
                # Run blocking generation in a thread so the event loop stays
                # responsive; the model is stateless per-request (no shared KV),
                # so no global lock is needed (lock starved all requests).
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()

                def _produce():
                    try:
                        for text in engine.stream(
                            prompt, max_new_tokens=req.max_tokens, temperature=req.temperature,
                            top_k=req.top_k, top_p=req.top_p,
                            repetition_penalty=req.repetition_penalty):
                            loop.call_soon_threadsafe(queue.put_nowait, ("data", text))
                    except Exception as exc:  # surface as SSE error, not silent cut
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

                import threading
                worker = threading.Thread(target=_produce, daemon=True)
                worker.start()
                while True:
                    if await request.is_disconnected():
                        return
                    kind, payload = await queue.get()
                    if kind == "done":
                        break
                    if kind == "error":
                        yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk",
                            "created": created, "model": req.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "error",
                                         "error": payload}]}) + "\n\n"
                        break
                    yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk", "created": created,
                        "model": req.model, "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}]}) + "\n\n"
                yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"

        if req.stream:
            return StreamingResponse(chunks(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        if await request.is_disconnected():
            raise HTTPException(499, "client disconnected")
        try:
            text = await asyncio.to_thread(
                engine.generate, prompt, req.max_tokens, req.temperature,
                req.top_k, req.top_p, req.repetition_penalty)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse({"id": request_id, "object": "chat.completion", "created": created, "model": req.model,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

    return app


def main():
    p = argparse.ArgumentParser(description="SmaulLinear server (local only unless --api-token is set)")
    p.add_argument("--model", default="./runs/linear")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    p.add_argument("--architecture", default=None, choices=["rawr", "plain"],
                   help="Expected architecture (default: auto-detect from checkpoint)")
    p.add_argument("--embedding-storage", default=None, choices=["ram", "mmap"],
                   help="Embedding backend override (default: checkpoint's)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--api-token", default=None,
                   help="Require Bearer token (or SMAUL_API_TOKEN env) for /v1/chat/completions")
    p.add_argument("--max-prompt-tokens", type=int, default=MODEL_WINDOW,
                   help=f"max prompt tokens (default {MODEL_WINDOW} = model window)")
    args = p.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not (args.api_token or os.environ.get("SMAUL_API_TOKEN")):
        print(f"[WARN] binding to {args.host} without --api-token exposes open inference to the network")
    engine = LinearInference(args.model, args.device, args.dtype,
                             architecture=args.architecture,
                             embedding_storage=args.embedding_storage)
    uvicorn.run(create_app(engine, args.max_prompt_tokens, api_token=args.api_token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
