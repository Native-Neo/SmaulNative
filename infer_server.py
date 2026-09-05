#!/usr/bin/env python3
"""Local RWKV-X server with a Claude/Gemini-style chat UI."""

import argparse
import json
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from inference import RWKVXInference


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmaulNative</title>
<style>
:root{--bg:#101010;--panel:#171717;--panel2:#202020;--text:#f1f1f1;--muted:#9b9b9b;--line:#2d2d2d;--accent:#d6b4ff;--user:#262626}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;height:100vh;overflow:hidden}
.app{display:grid;grid-template-columns:260px 1fr;height:100vh}.side{border-right:1px solid var(--line);background:#141414;padding:14px;display:flex;flex-direction:column;gap:12px}.brand{font-weight:700;font-size:18px;padding:8px}.new{border:1px solid var(--line);background:var(--panel2);color:var(--text);border-radius:10px;padding:10px;cursor:pointer}.history{overflow:auto;display:flex;flex-direction:column;gap:4px}.history button{border:0;background:transparent;color:#ccc;text-align:left;padding:9px;border-radius:8px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.history button:hover{background:#242424}.sidefoot{margin-top:auto;color:var(--muted);font-size:12px;padding:8px}.main{min-width:0;display:flex;flex-direction:column}.top{height:58px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 20px}.model{font-weight:650}.status{color:var(--muted);font-size:12px}.chat{flex:1;overflow:auto}.messages{max-width:850px;margin:auto;padding:35px 22px 140px}.welcome{padding:70px 10px;text-align:center}.welcome h1{font-size:34px;margin:0 0 10px}.welcome p{color:var(--muted)}.msg{margin:25px 0}.role{font-size:12px;color:var(--muted);margin-bottom:7px}.bubble{line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere}.user .bubble{background:var(--user);border-radius:15px;padding:13px 16px}.assistant .bubble{padding-right:15px}.bubble code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.composer{position:fixed;bottom:0;left:260px;right:0;padding:18px 20px 22px;background:linear-gradient(transparent,var(--bg) 25%)}.box{max-width:850px;margin:auto;background:var(--panel);border:1px solid #363636;border-radius:17px;padding:10px;box-shadow:0 8px 35px #0008}.box textarea{width:100%;min-height:48px;max-height:180px;resize:none;background:transparent;border:0;outline:0;color:var(--text);padding:9px;font:inherit}.actions{display:flex;align-items:center;justify-content:space-between;padding:4px}.hint{font-size:11px;color:var(--muted)}.send{width:36px;height:36px;border:0;border-radius:11px;background:var(--accent);color:#17121c;font-weight:800;cursor:pointer}.send:disabled{opacity:.45;cursor:default}
@media(max-width:720px){.app{grid-template-columns:1fr}.side{display:none}.composer{left:0}.messages{padding-left:14px;padding-right:14px}.top{padding:0 14px}}
</style></head>
<body><div class="app"><aside class="side"><div class="brand">SmaulNative</div><button class="new" onclick="newChat()">＋ New chat</button><div id="history" class="history"></div><div class="sidefoot">RWKV-X local inference<br><span id="device"></span></div></aside><main class="main"><header class="top"><div class="model">RWKV-X <span id="modelmeta" class="status"></span></div><div id="status" class="status">Ready</div></header><section class="chat" id="chat"><div id="messages" class="messages"><div class="welcome"><h1>How can I help?</h1><p>Your conversation stays on this machine.</p></div></div></section><div class="composer"><div class="box"><textarea id="input" placeholder="Message SmaulNative…" rows="1"></textarea><div class="actions"><span class="hint">Enter to send · Shift+Enter for newline</span><button id="send" class="send" onclick="send()">↑</button></div></div></div></main></div>
<script>
let messages=[],chats=JSON.parse(localStorage.getItem('smaul-chats')||'[]'),busy=false;const $=id=>document.getElementById(id),input=$('input');
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}function render(s){let x=esc(s);x=x.replace(/```([\s\S]*?)```/g,'<pre style="background:#0b0b0b;padding:14px;border-radius:10px;overflow:auto"><code>$1</code></pre>');x=x.replace(/`([^`]+)`/g,'<code style="background:#252525;padding:2px 5px;border-radius:5px">$1</code>');x=x.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');return x.replace(/\n/g,'<br>')}
function draw(){let m=$('messages');if(!messages.length){m.innerHTML='<div class="welcome"><h1>How can I help?</h1><p>Your conversation stays on this machine.</p></div>';return}m.innerHTML=messages.map(x=>`<div class="msg ${x.role}"><div class="role">${x.role==='user'?'You':'SmaulNative'}</div><div class="bubble">${render(x.content)}</div></div>`).join('');$('chat').scrollTop=$('chat').scrollHeight}
function save(){localStorage.setItem('smaul-chats',JSON.stringify(chats.slice(-30)));localStorage.setItem('smaul-current',JSON.stringify(messages))}function history(){let h=$('history');h.innerHTML=chats.map((c,i)=>`<button onclick="loadChat(${i})">${esc(c.title||'New chat')}</button>`).join('')}function loadChat(i){messages=chats[i].messages||[];draw();save()}function newChat(){messages=[];draw();save();input.focus()}
async function send(){if(busy)return;let text=input.value.trim();if(!text)return;input.value='';input.style.height='48px';messages.push({role:'user',content:text});messages.push({role:'assistant',content:''});draw();busy=true;$('send').disabled=true;$('status').textContent='Thinking…';let started=performance.now();try{let r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'rwkv-x',messages:messages.slice(0,-1),stream:true,max_tokens:256,temperature:.7,top_p:.95})});if(!r.ok)throw new Error(await r.text());let reader=r.body.getReader(),dec=new TextDecoder(),buf='';while(true){let q=await reader.read();if(q.done)break;buf+=dec.decode(q.value,{stream:true});let parts=buf.split('\n');buf=parts.pop();for(let line of parts){if(!line.startsWith('data: '))continue;let d=line.slice(6);if(d==='[DONE]')continue;try{let j=JSON.parse(d),c=j.choices?.[0]?.delta?.content||'';messages[messages.length-1].content+=c;draw()}catch(e){}}}let sec=(performance.now()-started)/1000;$('status').textContent='Ready · '+((messages[messages.length-1].content.length/4)/Math.max(sec,.01)).toFixed(1)+' tok/s';let title=messages.find(x=>x.role==='user')?.content?.slice(0,42)||'New chat';chats.push({title,messages:[...messages]});save();history()}catch(e){messages[messages.length-1].content='Error: '+e.message;draw();$('status').textContent='Error'}finally{busy=false;$('send').disabled=false}}
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,180)+'px'});try{let old=JSON.parse(localStorage.getItem('smaul-current'));if(Array.isArray(old))messages=old}catch(e){}history();draw();fetch('/health').then(r=>r.json()).then(x=>{$('device').textContent=x.device;$('modelmeta').textContent=x.parameters+' params'}).catch(()=>{});
</script></body></html>'''


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "rwkv-x"
    messages: List[Message]
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0, le=5)
    top_p: float = Field(0.95, gt=0, le=1)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.05, ge=0.5, le=2)
    stream: bool = True
    system: Optional[str] = None


def create_app(engine: RWKVXInference):
    app = FastAPI(title="SmaulNative RWKV-X", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML

    @app.get("/health")
    async def health():
        return {"status":"ok","device":str(engine.device),"parameters":f"{engine.model.num_parameters()/1e6:.1f}M"}

    @app.get("/v1/models")
    async def models():
        return {"object":"list","data":[{"id":"rwkv-x","object":"model","owned_by":"SmaulNative"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        msgs=[m.model_dump() for m in req.messages]
        system=req.system or "You are SmaulNative, a helpful local AI assistant. Be concise, accurate, and practical."
        prompt=engine.chat_prompt(msgs, system);created=int(time.time());request_id="chatcmpl-"+uuid.uuid4().hex

        def chunks():
            for text in engine.stream(prompt,max_new_tokens=req.max_tokens,temperature=req.temperature,top_k=req.top_k,top_p=req.top_p,repetition_penalty=req.repetition_penalty):
                yield "data: "+json.dumps({"id":request_id,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{"content":text},"finish_reason":None}]} )+"\n\n"
            yield "data: "+json.dumps({"id":request_id,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})+"\n\n"
            yield "data: [DONE]\n\n"

        if req.stream:
            return StreamingResponse(chunks(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        text=engine.generate(prompt,max_new_tokens=req.max_tokens,temperature=req.temperature,top_k=req.top_k,top_p=req.top_p,repetition_penalty=req.repetition_penalty)
        return JSONResponse({"id":request_id,"object":"chat.completion","created":created,"model":req.model,"choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}]})

    return app


def main():
    p=argparse.ArgumentParser(description="SmaulNative RWKV-X server")
    p.add_argument("--model",default="./SmaulNative");p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"]);p.add_argument("--dtype",default="auto",choices=["auto","fp32","fp16","bf16"]);p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8080)
    args=p.parse_args();engine=RWKVXInference(args.model,args.device,args.dtype);uvicorn.run(create_app(engine),host=args.host,port=args.port)


if __name__=="__main__":main()
