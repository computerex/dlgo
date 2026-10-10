"""Strict comparison of two matched HTTP benchmark reports."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline')
p.add_argument('candidate')
p.add_argument('--output', required=True)
a = p.parse_args()
x, y = [json.loads(Path(name).read_text()) for name in (a.baseline, a.candidate)]
for key in ('request_sha256', 'model', 'readout', 'question_count'):
    if x[key] != y[key]:
        raise RuntimeError(f'mismatched {key}')
if len(x['runs']) != len(y['runs']):
    raise RuntimeError('sample counts differ')
changes = []
max_probability_delta = 0.0
for index, (old, new) in enumerate(zip(x['runs'], y['runs'])):
    if old['input_tokens'] != new['input_tokens']:
        raise RuntimeError(f'token count differs at {index}')
    for phase in ('cold', 'warm'):
        if new['degraded_questions'] or not new['answers_identical']:
            raise RuntimeError(f'candidate degraded/cache parity failed at {index}')
        for key, answer in old[phase + '_answers'].items():
            other = new[phase + '_answers'][key]
            for decision_key in ('type', 'choice', 'degraded'):
                if answer.get(decision_key) != other.get(decision_key):
                    changes.append({'run': index, 'phase': phase, 'question': key, 'field': decision_key})
            for choice, value in answer.get('probabilities', {}).items():
                max_probability_delta = max(max_probability_delta, abs(value - other['probabilities'][choice]))
result = {'baseline': x['summary'], 'candidate': y['summary'],
          'decision_changes': changes, 'max_probability_delta': max_probability_delta,
          'speedup_p50_cold': x['summary']['cold']['p50_ms']/y['summary']['cold']['p50_ms'],
          'speedup_p95_cold': x['summary']['cold']['p95_ms']/y['summary']['cold']['p95_ms']}
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
if changes:
    raise RuntimeError('matched-request decision parity failed')
