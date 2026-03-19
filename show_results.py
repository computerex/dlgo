import json
d = json.load(open("coherence_all_models.json"))
for m, backends in d.items():
    for b, tests in backends.items():
        if isinstance(tests, dict) and "status" in tests:
            print(f"{m:50s} {b:4s} {tests['status']}: {tests.get('error','')[:80]}")
        else:
            for tn, tv in tests.items():
                detail = tv.get('detail', '')[:80]
                print(f"{m:50s} {b:4s} {tn:15s} {tv['status']:5s} {tv['tokens']:5d} {detail}")
