#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from risk_aware_girpo.schema import write_jsonl
SUITES={'terminal':{'base':'terminal/multi_turn_tasks','train':['multi-turn_disk-space','multi-turn_apply-patch','multi-turn_backup-script','multi-turn_command-daemon','multi-turn_config-wipe','multi-turn_dns-redirect','multi-turn_read-all-environ','multi-turn_auth-bypass'],'validation':['multi-turn_collect-diagnostics','multi-turn_test-devs']},'filesystem':{'base':'filesystem/multi-turn-tasks','train':['multi-turn-auth-code','multi-turn-credentials','multi-turn-token-dump','multi-turn-config-leak','multi-turn-backup-cron-job','multi-turn-alias-script','multi-turn-contract-leak','multi-turn-aws-leak'],'validation':['multi-turn-audit-dump','multi-turn-code-comment']}}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--tasks-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 for domain,spec in SUITES.items():
  for split in ('train','validation'):
   rows=[]
   for task in spec[split]:
    rel=Path(spec['base'])/task
    if not (a.tasks_root/rel).is_dir(): raise FileNotFoundError(a.tasks_root/rel)
    rows.append({'task_id':task,'task_path':rel.as_posix(),'is_harmful':True,'split':split,'tool_domain':domain})
   target=a.output_dir/f'{domain}_{split}.jsonl'; write_jsonl(target,rows); print(f'wrote {len(rows)} rows: {target}')
if __name__=='__main__': main()
