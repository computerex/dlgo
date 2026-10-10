"""Run the chess battery against a served model (default: 9B)."""
import argparse
import json
from pathlib import Path

import chess_battery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8097')
    parser.add_argument('--readout', default='systemone-train/artifacts/qwen35-9b-q3-readout-v1-gpu.json')
    parser.add_argument('--model', default='qwen35-9b')
    args = parser.parse_args()

    source = Path('systemone-train/request-chess-multi.json')
    request = json.loads(source.read_text(encoding='utf-8'))
    request['model'] = args.model
    request['readout'] = str(Path(args.readout).resolve())
    target = Path('bench/systemone/chess-request.json')
    target.write_text(json.dumps(request, indent=2), encoding='utf-8')

    chess_battery.URL = args.url.rstrip('/') + '/v1/systemone'
    chess_battery.REQUEST = str(target)
    chess_battery.main()


if __name__ == '__main__':
    main()
