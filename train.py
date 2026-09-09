#!/usr/bin/env python3
import argparse,contextlib,json,os,signal,shutil,sys,time
from pathlib import Path
from typing import Optional

if "--cpu" in sys.argv:
    threads=str(os.environ.get("SMAUL_CPU_THREADS") or max(1,(os.cpu_count() or 2)//2))
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(k,threads)
    os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS","SSE4.2");os.environ.setdefault("TORCHINDUCTOR_CPP_WRAPPER","1");os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE","1");os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS","ATEN,CPP")
import torch
from torch.optim import Optimizer
from rwkv_x_core import RWKVXModel,RWKV_CMix_MoE
from dataset import load_tokenizer,tokenizer_vocab_size,PretrainStream,SFTDataset,iter_texts,discover_files
from tokenizer import train_tokenizer
from stream_data import stream_dataset
import qat

DEFAULT_TARGET_PARAMS=256_000_000;STOP_REQUESTED=False

def _sigint_handler(signum,frame):
    global STOP_REQUESTED;STOP_REQUESTED=True;print("\n[Ctrl-C] Stop requested. Finishing current step, then saving checkpoint.")
signal.signal(signal.SIGINT,_sigint_handler)

class Lion(Optimizer):
    def __init__(self,params,lr=1e-4,betas=(.9,.99),weight_decay=.01):
        if lr<=0:raise ValueError("lr must be > 0")
        super().__init__(params,dict(lr=lr,betas=betas,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=closure() if closure else None
        for group in self.param_groups:
            lr,(b1,b2),wd=group["lr"],group["betas"],group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:continue
                st=self.state[p]
                if not st:st["exp_avg"]=torch.zeros_like(p)
                avg=st["exp_avg"]
                if wd:p.mul_(1-lr*wd)
                update=avg.mul(b1).add(p.grad,alpha=1-b1)
                p.add_(update.sign(),alpha=-lr);avg.lerp_(p.grad,1-b2)
        return loss

def set_router_only_training(model,router_only):
    if not model.cfg.is_moe:raise ValueError("set_router_only_training requires cfg.is_moe=True")
    gates={id(p) for m in model.modules() if isinstance(m,RWKV_CMix_MoE) for p in m.gate.parameters()};n=0
    for p in model.parameters():
        p.requires_grad_(id(p) in gates if router_only else True);n+=p.numel() if p.requires_grad else 0
    return n

class ResumeState:
    def __init__(self):self.global_step=0;self.total_tokens=0;self.file_path:Optional[str]=None;self.record_index=0;self.epoch=0;self.buffer_tokens=[]
    @classmethod
    def load(cls,path):
        s=cls()
        if path.exists():
            try:
                d=json.loads(path.read_text());s.global_step=d.get("global_step",0);s.total_tokens=d.get("total_tokens",0);s.file_path=d.get("file_path");s.record_index=d.get("record_index",0);s.epoch=d.get("epoch",0);s.buffer_tokens=d.get("buffer_tokens",[])
            except Exception as e:print(f"[WARN] could not load resume state: {e}")
        return s
    def save(self,path):
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".json.tmp");tmp.write_text(json.dumps(self.__dict__,indent=2));os.replace(tmp,path)

def _save_rng_state(path):
    state={"torch":torch.get_rng_state()}
    if torch.cuda.is_available():state["cuda"]=torch.cuda.get_rng_state_all()
    tmp=path.with_suffix(".pt.tmp");torch.save(state,tmp);os.replace(tmp,path)

def _load_rng_state(path):
    if not path.exists():return
    try:
        state=torch.load(path,map_location="cpu",weights_only=False);torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:print(f"[WARN] could not restore RNG state: {e}")

def save_checkpoint(model,optimizer,resume,output_dir,checkpoint_dir,tokenizer_path,save_dtype="fp32",save_optimizer=True):
    output_dir.mkdir(parents=True,exist_ok=True);checkpoint_dir.mkdir(parents=True,exist_ok=True);model=getattr(model,"_orig_mod",model);model.save_pretrained(output_dir,dtype=save_dtype,include_upstream=False)
    bundled=output_dir/"tokenizer.json"
    if tokenizer_path.resolve()!=bundled.resolve():shutil.copy2(tokenizer_path,bundled)
    if save_optimizer:
        tmp=checkpoint_dir/"optimizer.pt.tmp";torch.save(optimizer.state_dict(),tmp);os.replace(tmp,checkpoint_dir/"optimizer.pt");_save_rng_state(checkpoint_dir/"rng_state.pt")
    resume.save(checkpoint_dir/"resume_state.json");print(f"[SAVE COMPLETE] {output_dir}")

def _autocast(args,device):
    if device.type=="cuda":return torch.autocast(device_type="cuda",dtype=torch.float16 if args.precision=="fp16" else torch.bfloat16)
    if args.precision=="bf16":return torch.autocast(device_type="cpu",dtype=torch.bfloat16)
    return contextlib.nullcontext()

def _optimizer_step(args,model,optimizer,xb,yb,device,scaler):
    optimizer.zero_grad(set_to_none=True)
    with _autocast(args,device):_,loss,_=model(xb,labels=yb)
    if not torch.isfinite(loss):print(f"[WARN] non-finite loss {loss.item()}, skipping step");return None
    if scaler:scaler.scale(loss).backward();scaler.unscale_(optimizer)
    else:loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=True)
    if scaler:scaler.step(optimizer);scaler.update()
    else:optimizer.step()
    return loss

def _remote_token_stream(name,tokenizer,ctx_len,resume):
    ds=None;file=None;record=0
    if resume.file_path and "::" in resume.file_path:
        ds,file=resume.file_path.split("::",1);record=resume.record_index
    buf=list(resume.buffer_tokens)
    for text,pos in stream_dataset(name,start_dataset=ds,start_file=file,start_record=record,with_position=True):
        ids=tokenizer.encode(text)+[tokenizer.eos_token_id];buf.extend(ids)
        while len(buf)>=ctx_len+1:
            chunk=buf[:ctx_len+1];del buf[:ctx_len];yield torch.tensor(chunk[:-1]),torch.tensor(chunk[1:]),f"{pos[0]}::{pos[1]}",pos[2],list(buf)

def train_pretrain(args,model,optimizer,resume,device,tokenizer,scaler):
    if args.stream_dataset!="none":stream=_remote_token_stream(args.stream_dataset,tokenizer,args.ctx_len,resume);remote=True
    else:
        if resume.file_path and not Path(resume.file_path).is_file():raise FileNotFoundError(f"resume dataset file no longer exists: {resume.file_path}")
        stream=PretrainStream(Path(args.dataset_dir),tokenizer,args.ctx_len,resume_file=resume.file_path,resume_record=resume.record_index,buffer_tokens=resume.buffer_tokens);remote=False
    model.train();batch_x=[];batch_y=[];t0=time.perf_counter();tok_since=0
    for item in stream:
        if remote:x,y,path,rec,buf=item
        else:x,y,(path,rec),buf=item,stream.buffer_tokens
        batch_x.append(x);batch_y.append(y)
        if len(batch_x)<args.batch_size:continue
        xb=torch.stack(batch_x).to(device);yb=torch.stack(batch_y).to(device);loss=_optimizer_step(args,model,optimizer,xb,yb,device,scaler)
        if loss is None:batch_x=[];batch_y=[];continue
        resume.global_step+=1;resume.total_tokens+=xb.numel();resume.file_path=path;resume.record_index=rec;resume.buffer_tokens=buf;tok_since+=xb.numel()
        if resume.global_step%args.log_every==0:
            dt=time.perf_counter()-t0;print(f"step {resume.global_step} | loss {loss.item():.4f} | {tok_since/max(dt,1e-9):.1f} tok/s | tokens {resume.total_tokens:,}");t0=time.perf_counter();tok_since=0
        batch_x=[];batch_y=[]
        if resume.global_step%args.save_every==0:save_checkpoint(model,optimizer,resume,Path(args.output_dir),Path(args.checkpoint_dir),Path(args.tokenizer_path),args.save_dtype,resume.global_step%args.optimizer_save_every==0)
        if STOP_REQUESTED:break

def train_sft(args,model,optimizer,resume,device,tokenizer,scaler):
    dataset=SFTDataset(Path(args.dataset_dir),tokenizer,args.ctx_len);model.train()
    for epoch in range(resume.epoch,args.epochs):
        perm=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(epoch)).tolist();start=resume.record_index if epoch==resume.epoch else 0;loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,sampler=perm[start:]);consumed=start
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device);loss=_optimizer_step(args,model,optimizer,xb,yb,device,scaler);consumed+=len(xb)
            if loss is None:continue
            resume.global_step+=1;resume.total_tokens+=xb.numel();resume.epoch,resume.record_index=epoch,consumed
            if resume.global_step%args.log_every==0:print(f"epoch {epoch} step {resume.global_step} | loss {loss.item():.4f}")
            if resume.global_step%args.save_every==0:save_checkpoint(model,optimizer,resume,Path(args.output_dir),Path(args.checkpoint_dir),Path(args.tokenizer_path),args.save_dtype,resume.global_step%args.optimizer_save_every==0)
            if STOP_REQUESTED:break
        if STOP_REQUESTED:break
        resume.epoch,resume.record_index=epoch+1,0

def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["pretrain","sft"],required=True);p.add_argument("--dataset_dir",default="./datasets");p.add_argument("--stream_dataset",choices=["none","hindi","english","openthoughts","all"],default="none");p.add_argument("--output_dir",default="./SmaulNative");p.add_argument("--checkpoint_dir",default="./SmaulNative");p.add_argument("--tokenizer_path",default="./SmaulNative/tokenizer.json");p.add_argument("--tokenizer_vocab_size",type=int,default=65536);p.add_argument("--tokenizer_max_records",type=int,default=5_000_000);p.add_argument("--target_params",type=int,default=DEFAULT_TARGET_PARAMS);p.add_argument("--n_embd",type=int,default=832);p.add_argument("--n_layer",type=int,default=17);p.add_argument("--head_size",type=int,default=64);p.add_argument("--n_moba_layer",type=int,default=3);p.add_argument("--ctx_len",type=int,default=1024);p.add_argument("--precision",choices=["fp32","fp16","bf16"],default=None);p.add_argument("--save_dtype",choices=["fp32","fp16","bf16"],default="fp32");p.add_argument("--batch_size",type=int,default=2);p.add_argument("--epochs",type=int,default=3);p.add_argument("--learning_rate",type=float,default=1e-4);p.add_argument("--optimizer",choices=["adafactor","lion","adamw"],default="adafactor");p.add_argument("--log_every",type=int,default=1);p.add_argument("--save_every",type=int,default=5000);p.add_argument("--optimizer_save_every",type=int,default=None);p.add_argument("--new_data",action="store_true");p.add_argument("--train_router_only",action="store_true");p.add_argument("--qat",action="store_true");p.add_argument("--qat_calib_batches",type=int,default=64);p.add_argument("--qat_export_dir",default=None);p.add_argument("--compile",action="store_true");p.add_argument("--cpu",action="store_true");a=p.parse_args();a.precision=a.precision or ("fp16" if torch.cuda.is_available() and not a.cpu else "fp32");a.optimizer_save_every=a.optimizer_save_every or a.save_every
    if a.target_params<=0 or a.tokenizer_max_records<0 or a.n_embd<=0 or a.head_size<=0 or a.n_layer<=0 or a.n_moba_layer<0 or a.n_moba_layer>=a.n_layer:p.error("invalid model/training parameters")
    if a.n_embd%a.head_size:p.error("--n_embd must be divisible by --head_size")
    if a.cpu and a.precision=="fp16":p.error("--precision fp16 requires CUDA")
    if a.mode=="sft" and a.stream_dataset!="none":p.error("--stream_dataset is supported for pretraining only")
    return a

def build_model(args,tokenizer):
    out=Path(args.output_dir)
    if (out/"config.json").exists() and (out/"model.safetensors").exists():return RWKVXModel.from_pretrained(out)
    from rwkv_x_core import RWKVXConfig
    cfg=RWKVXConfig(vocab_size=tokenizer_vocab_size(tokenizer),n_embd=args.n_embd,n_layer=args.n_layer,n_moba_layer=args.n_moba_layer,head_size=args.head_size);cfg.ctx_len_hint=args.ctx_len
    return RWKVXModel(cfg)

def main():
    args=parse_args();device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu");print(f"[DEVICE] {device} | precision={args.precision}")
    if args.cpu:
        from cpu import configure;print(f"[CPU] {configure()} threads, native WKV, compile={args.compile}")
    tokenizer_path=Path(args.tokenizer_path);out=Path(args.output_dir);bundled=out/"tokenizer.json"
    if (out/"config.json").exists() and bundled.exists() and bundled.resolve()!=tokenizer_path.resolve():tokenizer_path=bundled
    if not tokenizer_path.exists():
        print(f"[TOKENIZER] training {tokenizer_path}");train_tokenizer(Path(args.dataset_dir),tokenizer_path,args.tokenizer_vocab_size,args.stream_dataset,args.tokenizer_max_records)
    tokenizer=load_tokenizer(tokenizer_path);model=build_model(args,tokenizer).to(device)
    if args.train_router_only:print(f"[ROUTER-ONLY] {set_router_only_training(model,True):,} trainable params")
    if args.qat:
        n=qat.prepare_qat(model);print(f"[QAT] fake-quantizing {n} linears")
        if args.stream_dataset:calib_texts=(text for text,_ in stream_dataset(args.stream_dataset))
        else:calib_texts=(text for text,_path,_idx in iter_texts(discover_files(Path(args.dataset_dir))))
        done=qat.calibrate(model,tokenizer,calib_texts,args.ctx_len,device,args.qat_calib_batches);print(f"[QAT] calibrated {done} batches")
    if args.compile:model=torch.compile(model,mode="max-autotune")
    checkpoint_dir=Path(args.checkpoint_dir);resume=ResumeState.load(checkpoint_dir/"resume_state.json")
    if args.new_data:resume=ResumeState()
    else:_load_rng_state(checkpoint_dir/"rng_state.pt")
    opt_map={"lion":Lion,"adamw":torch.optim.AdamW,"adafactor":torch.optim.Adafactor}
    if args.cpu and args.optimizer=="lion":
        from cpu import NativeLion;opt_map["lion"]=NativeLion
    optimizer=opt_map[args.optimizer](model.parameters(),lr=args.learning_rate);opt_path=checkpoint_dir/"optimizer.pt"
    if opt_path.exists() and not args.new_data:
        try:optimizer.load_state_dict(torch.load(opt_path,map_location="cpu",weights_only=False))
        except Exception as e:print(f"[WARN] could not restore optimizer: {e}")
    scaler=torch.amp.GradScaler("cuda") if device.type=="cuda" and args.precision=="fp16" else None
    try:
        train_pretrain(args,model,optimizer,resume,device,tokenizer,scaler) if args.mode=="pretrain" else train_sft(args,model,optimizer,resume,device,tokenizer,scaler)
    finally:save_checkpoint(model,optimizer,resume,Path(args.output_dir),checkpoint_dir,tokenizer_path,args.save_dtype,True)
    if args.qat and args.qat_export_dir:
        import copy
        exported=copy.deepcopy(getattr(model,"_orig_mod",model)).cpu();n=qat.convert_qat(exported);exported.save_pretrained(Path(args.qat_export_dir));shutil.copy2(tokenizer_path,Path(args.qat_export_dir)/"tokenizer.json");print(f"[QAT] converted {n} linears")

if __name__=="__main__":main()
