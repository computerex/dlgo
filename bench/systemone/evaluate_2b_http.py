"""Run existing Yu-Gi-Oh scorers against a separately served 2B model."""
import argparse
import json
from pathlib import Path
import yugioh_battery
import yugioh_generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8097')
    parser.add_argument('--readout', default='systemone-train/artifacts/qwen38-2b-q4-readout-v1-gpu.json')
    parser.add_argument('--model', default='qwen38-2b')
    parser.add_argument('--generation', action='store_true')
    args = parser.parse_args()
    source = Path('systemone-train/request-yugioh-multi.json')
    request = json.loads(source.read_text(encoding='utf-8'))
    request['model'] = args.model
    request['readout'] = str(Path(args.readout).resolve())
    target = Path('bench/systemone/2b-yugioh-request.json')
    target.write_text(json.dumps(request, indent=2), encoding='utf-8')
    yugioh_battery.URL = args.url.rstrip('/') + '/v1/systemone'
    yugioh_battery.REQUEST = str(target)
    yugioh_battery.main()
    if args.generation:
        yugioh_generation.URL = args.url.rstrip('/') + '/v1/chat/completions'
        yugioh_generation.MODEL = args.model
        yugioh_generation.run(False, 512)


if __name__ == '__main__':
    main()
