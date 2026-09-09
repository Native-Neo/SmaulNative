#!/usr/bin/env python3
"""Stream HF Parquet records directly without downloading datasets to disk."""
from __future__ import annotations
import argparse,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor
from typing import Any,Dict,Iterator
import pyarrow.parquet as pq
from huggingface_hub import HfApi,HfFileSystem
from filter_data import filter_text

DATASETS: Dict[str,Dict[str,str]]={
    "hindi":{"repo_id":"HuggingFaceFW/fineweb-2","path":"data/hin_Deva/train"},
    "english":{"repo_id":"HuggingFaceFW/fineweb","path":"data/100BT"},
    "openthoughts":{"repo_id":"open-thoughts/OpenThoughts3-1.2M","path":"data"},
}
HF_TOKEN=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
api=HfApi(token=HF_TOKEN);fs=HfFileSystem(token=HF_TOKEN)

def _conversation(value: Any)->str:
    if not isinstance(value,list):return ""
    parts=[]
    for turn in value:
        if not isinstance(turn,dict):continue
        text=turn.get("value")
        if not isinstance(text,str) or not text.strip():continue
        role=str(turn.get("from","")).strip();parts.append(f"{role}: {text.strip()}" if role else text.strip())
    return "\n".join(parts)

def _files(repo_id:str,path:str)->list[str]:
    items=api.list_repo_tree(repo_id=repo_id,repo_type="dataset",path_in_repo=path,recursive=True)
    return sorted(item.path for item in items if getattr(item,"path","").endswith(".parquet"))

def _text_column(pf:pq.ParquetFile)->tuple[str|None,bool]:
    names=pf.schema_arrow.names;lower={name.lower():name for name in names}
    for name in ("text","content","document","body"):
        if name in lower:return lower[name],False
    if "conversations" in lower:return lower["conversations"],True
    return None,False

def _read_row_group(remote:str,row_group:int,columns:list[str]|None,token:str|None)->list[Any]:
    retries=max(1,int(os.environ.get("SMAUL_STREAM_RETRIES","6")))
    for attempt in range(retries):
        try:
            with HfFileSystem(token=token).open(remote,"rb") as handle:
                return pq.ParquetFile(handle).read_row_group(row_group,columns=columns).to_pylist()
        except Exception as exc:
            if attempt+1>=retries:raise
            delay=min(30.0,2.0**attempt);print(f"[STREAM] row group {row_group} read failed: {type(exc).__name__}; retrying in {delay:.0f}s",file=sys.stderr);time.sleep(delay)

def _stream_file(config:Dict[str,str],dataset_name:str,rel_path:str,min_chars:int,max_chars:int,skip:int,with_position:bool,workers:int)->Iterator[Any]:
    remote=f"datasets/{config['repo_id']}/{rel_path}"
    with fs.open(remote,"rb") as handle:
        pf=pq.ParquetFile(handle);column,conversation=_text_column(pf);columns=[column] if column else None;row_groups=pf.num_row_groups
    record=0;batch_size=max(1,workers*2);token=HF_TOKEN
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0,row_groups,batch_size):
            futures=[pool.submit(_read_row_group,remote,i,columns,token) for i in range(first,min(first+batch_size,row_groups))]
            for future in futures:
                for value in future.result():
                    if record<skip:record+=1;continue
                    if column:text=_conversation(value) if conversation else value
                    else:
                        text=max((item for item in value.values() if isinstance(item,str)),key=len,default="") if isinstance(value,dict) else ""
                    position=(dataset_name,rel_path,record+1);record+=1;text=filter_text(text,dataset_name,min_chars,max_chars)
                    if text is not None:yield (text,position) if with_position else text

def stream_dataset(name:str,min_chars:int=20,max_chars:int=1_000_000,start_dataset:str|None=None,start_file:str|None=None,start_record:int=0,with_position:bool=False,workers:int|None=None)->Iterator[Any]:
    names=list(DATASETS) if name=="all" else [name]
    if name!="all" and name not in DATASETS:raise ValueError(f"unknown dataset: {name}")
    workers=workers or int(os.environ.get("SMAUL_STREAM_WORKERS","0"));workers=min(4,max(1,os.cpu_count() or 1)) if workers<=0 else workers
    active_dataset=start_dataset is None
    for dataset_name in names:
        if not active_dataset:
            if dataset_name!=start_dataset:continue
            active_dataset=True
        config=DATASETS[dataset_name];paths=_files(config["repo_id"],config["path"])
        if not paths:raise RuntimeError(f"No Parquet files found for {dataset_name}")
        active_file=start_file is None or dataset_name!=start_dataset
        for rel_path in paths:
            if not active_file:
                if rel_path!=start_file:continue
                active_file=True
            skip=start_record if dataset_name==start_dataset and rel_path==start_file else 0
            print(f"[STREAM] {dataset_name}/{rel_path}"+(f" from row {skip:,}" if skip else ""),file=sys.stderr)
            yield from _stream_file(config,dataset_name,rel_path,min_chars,max_chars,skip,with_position,workers)

def main()->None:
    p=argparse.ArgumentParser(description="Stream and filter HF datasets with no dataset files on disk");p.add_argument("--dataset",choices=[*DATASETS,"all"],default="all");p.add_argument("--min_chars",type=int,default=20);p.add_argument("--max_chars",type=int,default=1_000_000);p.add_argument("--max_records",type=int,default=0);p.add_argument("--workers",type=int,default=None);args=p.parse_args();count=0
    for text in stream_dataset(args.dataset,args.min_chars,args.max_chars,workers=args.workers):
        print(json.dumps({"text":text},ensure_ascii=False),flush=True);count+=1
        if args.max_records and count>=args.max_records:print(f"[DONE] streamed {count:,} records",file=sys.stderr);return
    print(f"[DONE] streamed {count:,} records",file=sys.stderr)

if __name__=="__main__":main()
