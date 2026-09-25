#!/usr/bin/env python3
"""Local SmaulLinear server with chat UI."""

import argparse
import json
import threading
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from inference import LinearInference

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmaulLinear</title><style>
body{margin:0;background:#101010;color:#f1f1f1;font:15px system-ui,sans-serif;height:100vh;overflow:hidden}.app{display:grid;grid-template-columns:260px 1fr;height:100vh}.side{border-right:1px solid #2d2d2d;background:#141414;padding:14px;display:flex;flex-direction:column;gap:12px}.brand{font-weight:700;font-size:18px;padding:8px}.new{border:1px solid #2d2d2d;background:#202020;color:#f1f1f1;border-radius:10px;padding:10px;cursor:pointer}.history{overflow:auto;display:flex;flex-direction:column;gap:4px}.main{min-width:0;display:flex;flex-direction:column}.top{height:58px;border-bottom:1px solid #2d2d2d;display:flex;align-items:center;justify-content:space-between;padding:0 20px}.chat{flex:1;overflow:auto}.messages{max-width:850px;margin:auto;padding:35px 22px 140px}.msg{margin:25px 0}.role{font-size:12px;color:#9b9b9b;margin-bottom:7px}.bubble{line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere}.composer{position:fixed;bottom:0;left:260px;right:0;padding:18px 20px 22px}.box{max-width:850px;margin:auto;background:#171717;border:1px solid #2d2d2d;border-radius:14px;padding:12px}
</style></head><body><div class="app"><aside class="side"><div class="brand">SmaulLinear</div><button class="new" onclick="newChat()">New chat</button><div id="history" class="history"></div></aside><main class="main"><header class="top"><div>Linear <span id="modelmeta"></span></div><div id="status">Ready</div></header><section class="chat" id="chat"><div id="messages" class="messages"></div></section><div class="composer"><div class="box"><textarea id="input" placeholder="Message SmaulLinear…" rows="1" style="width:100%"></textarea><button id="send" onclick="send()">Send</button></div></div></main></div><script>
let messages=[],busy=false;const $=id=>document.getElementById(id),input=$('input');function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function draw(){let m=$('messages');m.innerHTML=messages.map(x=>`<div class="msg"><div class="role">${esc(x.role)}</div><div class="bubble">${esc(x.content)}</div></div>`).join('');$('chat').scrollTop=$('chat').scrollHeight}async function send(){if(busy)return;let text=input.value.trim();if(!text)return;input.value='';messages.push({role:'user',content:text});messages.push({role:'assistant',content:''});draw();busy=true;try{let r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'smaul-linear',messages:messages.slice(0,-1),stream:true,max_tokens:256})});let rd=r.body.getReader(),dec=new TextDecoder(),buf='';while(true){let {done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let parts=buf.split('\n\n');buf=parts.pop();for(let p of parts){if(p.startsWith('data: ')){let d=p.slice(6);if(d==='[DONE]')continue;try{messages[messages.length-1].content+=JSON.parse(d).choices[0].delta.content||'';draw()}catch(e){}}}}catch(e){messages[messages.length-1].content+='[error]'}busy=false;draw()}function newChat(){messages=[];draw()}
</script></body></html>'''


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "smaul-linear"
    messages: List[Message]
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0, le=5)
    top_p: float = Field(0.95, gt=0, le=1)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.05, ge=0.5, le=2)
    stream: bool = True
    system: Optional[str] = None


def create_app(engine: LinearInference, max_prompt_tokens: int = 4096):
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")
    app = FastAPI(title="SmaulLinear", version="0.2.0")
    model_lock = threading.Lock()

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
    async def chat(req: ChatRequest):
        msgs = [m.model_dump() for m in req.messages]
        system = req.system or "You are SmaulLinear, a helpful local AI assistant. Be concise, accurate, and practical."
        prompt = engine.chat_prompt(msgs, system)
        prompt_tokens = len(engine.encode(prompt))
        if prompt_tokens > max_prompt_tokens:
            raise HTTPException(413, f"prompt exceeds {max_prompt_tokens} tokens ({prompt_tokens})")
        created = int(time.time())
        request_id = "chatcmpl-" + uuid.uuid4().hex

        def chunks():
            with model_lock:
                for text in engine.stream(prompt, max_new_tokens=req.max_tokens, temperature=req.temperature,
                                          top_k=req.top_k, top_p=req.top_p, repetition_penalty=req.repetition_penalty):
                    yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk", "created": created,
                        "model": req.model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}) + "\n\n"
                yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
                yield "data: [DONE]\n\n"

        if req.stream:
            return StreamingResponse(chunks(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        with model_lock:
            text = engine.generate(prompt, max_new_tokens=req.max_tokens, temperature=req.temperature,
                                   top_k=req.top_k, top_p=req.top_p, repetition_penalty=req.repetition_penalty)
        return JSONResponse({"id": request_id, "object": "chat.completion", "created": created, "model": req.model,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

    return app


def main():
    p = argparse.ArgumentParser(description="SmaulLinear server")
    p.add_argument("--model", default="./runs/linear")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    args = p.parse_args()
    engine = LinearInference(args.model, args.device, args.dtype)
    uvicorn.run(create_app(engine, args.max_prompt_tokens), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
