#!/usr/bin/env python3
"""Inspect option keys per ordinal candidate family."""
import json, collections

fams = collections.defaultdict(set)
with open('systemone-train/data/generated/splits-gp2/train.jsonl', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        fam = row.get('family')
        if fam in ('severity', 'priority', 'reply_urgency', 'satisfaction', 'risk', 'commitment',
                   'task_state', 'data_state', 'email_action', 'channel', 'tone'):
            crit = row['criteria']
            if isinstance(crit, str):
                crit = json.loads(crit)
            fams[fam].update(crit.keys())

for fam in sorted(fams):
    print(fam, '->', sorted(fams[fam]))
