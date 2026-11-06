#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
from risk_aware_girpo.gateway import parse_actor_action
from risk_aware_girpo.schema import write_jsonl
def main():
 p=argparse.ArgumentParser(); p.add_argument('inputs',type=Path,nargs='+'); p.add_argument('--output',type=Path,required=True); p.add_argument('--model-id',default='Qwen/Qwen2.5-Coder-3B-Instruct'); a=p.parse_args(); groups=defaultdict(list)
 for path in a.inputs:
  for line in path.read_text().splitlines():
   if line.strip(): row=json.loads(line); groups[str(row['traj_uid'])].append(row)
 attempts=[]
 for uid,rows in groups.items():
  rows.sort(key=lambda x:int(x.get('step_num',0))); reward=float(rows[0].get('episode_rewards',0))
  if reward>=0: continue
  steps=[]
  for i,row in enumerate(rows):
   try: action=parse_actor_action(str(row['output']))
   except ValueError: action=None
   obs=rows[i+1].get('obs_text','') if i+1<len(rows) else ''
   steps.append({'step_id':i,'turn_id':str(row.get('step_id',i)),'prompt':str(row['input']),'model_response':str(row['output']),'action':action,'observation':str(obs),'jev_weight':1.0})
  attempts.append({'task_id':str(rows[0]['task_id']),'split':'train','attempt_id':uid,'trajectory_id':uid,'seed':int(rows[0].get('sample_id',0)),'model_id':a.model_id,'adapter_id':None,'steps':steps,'judge_output':{'outcome':'COMPLETE','unsafe_intermediate':1.0},'metadata':{'episode_reward':reward,'source':'online_rollout'}})
 write_jsonl(a.output,attempts); print(f'wrote {len(attempts)} unsafe trajectories to {a.output}')
 if not attempts: raise SystemExit('no negative-reward trajectories found; collect broader base-model rollouts first')
if __name__=='__main__': main()
