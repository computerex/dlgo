#!/usr/bin/env python3
"""Run a sealed, ground-truth System One noul comparison: Jev 1.13 vs dlgo 9B.

All local noul calls include the same true/false semantic criteria sent to Jev.
The suite deliberately excludes the Charizard/Water parity demonstration; that
question remains a separate sealed regression. Results contain no credentials.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "systemone-train" / ".env"
OUT = ROOT / "bench" / "results" / "jev-noul-suite-comparison.json"
LOCAL_URL = "http://127.0.0.1:8080/v1/systemone"
READOUT = str(ROOT / "systemone-train" / "artifacts" / "qwen35-9b-q3-readout-v2-uni.json")

# Independently checkable facts/rules. No Charizard/Water pair or paraphrase.
CASES = [
    {"id": "moltres_water", "domain": "pokemon", "truth": True,
     "state": "A standard Pokemon type chart applies. Moltres is a Fire/Flying-type Pokemon.",
     "instructions": "Is Moltres, a Fire/Flying-type, weak to Water-type attacks?",
     "true": "Water attacks deal super-effective damage to Moltres.",
     "false": "Water attacks deal neutral, reduced, or no damage to Moltres."},
    {"id": "eevee_psychic", "domain": "pokemon", "truth": False,
     "state": "A standard Pokemon type chart applies. Eevee is a Normal-type Pokemon.",
     "instructions": "Is Eevee, a Normal-type, weak to Psychic-type attacks?",
     "true": "Psychic attacks deal super-effective damage to Eevee.",
     "false": "Psychic attacks deal neutral, reduced, or no damage to Eevee."},
    {"id": "pidgeot_ground", "domain": "pokemon", "truth": False,
     "state": "A standard Pokemon type chart applies. Pidgeot is a Normal/Flying-type Pokemon.",
     "instructions": "Is Pidgeot, a Normal/Flying-type, weak to Ground-type attacks?",
     "true": "Ground attacks deal super-effective damage to Pidgeot.",
     "false": "Ground attacks deal neutral, reduced, or no damage to Pidgeot."},
    {"id": "vaporeon_electric", "domain": "pokemon", "truth": True,
     "state": "A standard Pokemon type chart applies. Vaporeon is a Water-type Pokemon.",
     "instructions": "Is Vaporeon, a Water-type, weak to Electric-type attacks?",
     "true": "Electric attacks deal super-effective damage to Vaporeon.",
     "false": "Electric attacks deal neutral, reduced, or no damage to Vaporeon."},
    {"id": "arithmetic", "domain": "arithmetic", "truth": True,
     "state": "A shipment contains 17 crates with 24 items in each crate.",
     "instructions": "Does the shipment contain 408 items?",
     "true": "17 multiplied by 24 equals 408.",
     "false": "17 multiplied by 24 does not equal 408."},
    {"id": "leap_year", "domain": "calendar", "truth": True,
     "state": "Gregorian calendar rules apply. A year divisible by 4 is a leap year, except century years must also be divisible by 400.",
     "instructions": "Is the year 2024 a leap year?",
     "true": "2024 is a leap year with February 29.",
     "false": "2024 is not a leap year."},
    {"id": "chess_bishop_color", "domain": "chess", "truth": False,
     "state": "Standard international chess rules apply.",
     "instructions": "Can a bishop change the color of squares it occupies during a standard chess game?",
     "true": "A bishop can move from a light square to a dark square or vice versa.",
     "false": "A bishop always remains on squares of its original color."},
    {"id": "gdpr_erasure", "domain": "document", "truth": True,
     "state": "The GDPR is an EU regulation. Article 17 provides individuals a right to erasure of personal data in defined circumstances. Article 20 provides a right to data portability.",
     "instructions": "Does the stated GDPR text grant a right to erasure?",
     "true": "The stated text explicitly grants a right to erasure.",
     "false": "The stated text does not grant a right to erasure."},

    # Explicit rules and policy application.
    {"id": "policy_refund", "domain": "policy", "truth": True,
     "state": "Refund policy: a customer is eligible for a full refund when they cancel within 30 days of purchase and have used fewer than 10 service hours. Maya purchased 12 days ago and has used 4 service hours.",
     "instructions": "Is Maya eligible for a full refund under the stated policy?",
     "true": "Maya meets both the 30-day and fewer-than-10-hours requirements.",
     "false": "Maya fails at least one stated refund requirement."},
    {"id": "policy_refund_denied", "domain": "policy", "truth": False,
     "state": "Refund policy: a customer is eligible for a full refund when they cancel within 30 days of purchase and have used fewer than 10 service hours. Omar purchased 12 days ago and has used 14 service hours.",
     "instructions": "Is Omar eligible for a full refund under the stated policy?",
     "true": "Omar meets both the 30-day and fewer-than-10-hours requirements.",
     "false": "Omar fails at least one stated refund requirement."},
    {"id": "api_delete_confirmation", "domain": "tool_safety", "truth": False,
     "state": "Tool policy: permanently deleting a production database requires an explicit confirmation phrase from the user: DELETE PRODUCTION DATABASE. The user said: 'Please clean up old test records if it is safe.'",
     "instructions": "May the system permanently delete the production database now?",
     "true": "The user supplied the exact required confirmation phrase.",
     "false": "The user did not supply the exact required confirmation phrase."},

    # Quantitative and logical reasoning.
    {"id": "percentage_discount", "domain": "arithmetic", "truth": True,
     "state": "A jacket costs $80. It is discounted by 25%, then sales tax is ignored.",
     "instructions": "Is the discounted price $60?",
     "true": "Twenty-five percent of $80 is $20, leaving $60.",
     "false": "The stated discount does not result in a $60 price."},
    {"id": "percentage_discount_false", "domain": "arithmetic", "truth": False,
     "state": "A jacket costs $80. It is discounted by 25%, then sales tax is ignored.",
     "instructions": "Is the discounted price $55?",
     "true": "The stated discount results in a $55 price.",
     "false": "The stated discount does not result in a $55 price."},
    {"id": "syllogism", "domain": "logic", "truth": True,
     "state": "All licensed pilots have completed flight training. Elena is a licensed pilot.",
     "instructions": "Must Elena have completed flight training?",
     "true": "Elena is in a group for which completed flight training is stated to be required.",
     "false": "The statements do not establish that Elena completed flight training."},
    {"id": "syllogism_converse", "domain": "logic", "truth": False,
     "state": "All licensed pilots have completed flight training. Elena has completed flight training.",
     "instructions": "Must Elena be a licensed pilot?",
     "true": "Completing flight training is stated to be sufficient to make someone a licensed pilot.",
     "false": "The statements do not establish that completing training makes Elena a licensed pilot."},

    # Grounded document entailment and contradiction.
    {"id": "incident_severity", "domain": "document", "truth": True,
     "state": "Incident report: At 09:14 UTC the payment API returned HTTP 500 for 18% of checkout attempts. At 09:27 UTC engineers rolled back release 4.2. At 09:31 UTC errors returned to baseline. No customer data was lost.",
     "instructions": "Does the report state that rolling back release 4.2 occurred before errors returned to baseline?",
     "true": "The rollback at 09:27 UTC precedes the return to baseline at 09:31 UTC.",
     "false": "The report states that errors returned to baseline before the rollback."},
    {"id": "incident_data_loss", "domain": "document", "truth": False,
     "state": "Incident report: At 09:14 UTC the payment API returned HTTP 500 for 18% of checkout attempts. At 09:27 UTC engineers rolled back release 4.2. At 09:31 UTC errors returned to baseline. No customer data was lost.",
     "instructions": "Does the report state that customer data was lost?",
     "true": "The report states that customer data was lost.",
     "false": "The report explicitly states that no customer data was lost."},

    # Date/rule mechanics beyond the initial leap-year check.
    {"id": "century_leap", "domain": "calendar", "truth": False,
     "state": "Gregorian calendar rules apply. A year divisible by 4 is a leap year, except a year divisible by 100 is not unless it is also divisible by 400.",
     "instructions": "Is the year 1900 a leap year?",
     "true": "1900 satisfies the Gregorian leap-year requirements.",
     "false": "1900 is divisible by 100 but not by 400, so it is not a leap year."},
    {"id": "century_leap_2000", "domain": "calendar", "truth": True,
     "state": "Gregorian calendar rules apply. A year divisible by 4 is a leap year, except a year divisible by 100 is not unless it is also divisible by 400.",
     "instructions": "Is the year 2000 a leap year?",
     "true": "2000 is divisible by 400 and therefore satisfies the exception.",
     "false": "2000 does not satisfy the Gregorian leap-year requirements."},

    # Operations and data handling.
    {"id": "inventory_reorder", "domain": "operations", "truth": True,
     "state": "Inventory policy: reorder an item when available stock is below its reorder point. Widget A has 18 available units and a reorder point of 20 units.",
     "instructions": "Should Widget A be reordered under the stated policy?",
     "true": "18 available units is below the reorder point of 20.",
     "false": "18 available units is not below the reorder point of 20."},
    {"id": "inventory_reorder_no", "domain": "operations", "truth": False,
     "state": "Inventory policy: reorder an item when available stock is below its reorder point. Widget B has 20 available units and a reorder point of 20 units.",
     "instructions": "Should Widget B be reordered under the stated policy?",
     "true": "20 available units is below the reorder point of 20.",
     "false": "20 available units is not below the reorder point of 20."},
    {"id": "csv_duplicate", "domain": "data", "truth": True,
     "state": "CSV rows: order_id, customer\n1001, Ada\n1002, Ben\n1001, Cara\n1003, Dana. The order_id column must be unique.",
     "instructions": "Does this CSV violate the stated uniqueness requirement?",
     "true": "Order ID 1001 appears in more than one row.",
     "false": "Every order ID appears exactly once."},
    {"id": "set_membership", "domain": "data", "truth": False,
     "state": "The approved deployment regions are us-east-1, eu-west-1, and ap-southeast-2. The requested deployment region is us-west-2.",
     "instructions": "Is the requested deployment region approved?",
     "true": "us-west-2 is one of the approved deployment regions.",
     "false": "us-west-2 is not one of the approved deployment regions."},

    # Security and access control with fully supplied policy state.
    {"id": "access_control", "domain": "security", "truth": True,
     "state": "Access policy: users in the finance-admin group may read quarterly_financials. Priya belongs to finance-admin. The resource requested is quarterly_financials.",
     "instructions": "May Priya read the requested resource under the stated policy?",
     "true": "Priya is in the required finance-admin group for quarterly_financials.",
     "false": "Priya does not meet the stated group requirement."},
    {"id": "access_control_deny", "domain": "security", "truth": False,
     "state": "Access policy: users in the finance-admin group may read quarterly_financials. Leo belongs to finance-viewer. The resource requested is quarterly_financials.",
     "instructions": "May Leo read the requested resource under the stated policy?",
     "true": "Leo is in the required finance-admin group for quarterly_financials.",
     "false": "Leo does not meet the stated group requirement."},
    {"id": "mfa_policy", "domain": "security", "truth": False,
     "state": "Login policy: administrative access requires a valid password and a completed MFA challenge. Nina supplied a valid password but did not complete MFA.",
     "instructions": "Should Nina receive administrative access now?",
     "true": "Nina has completed every stated administrative-access requirement.",
     "false": "Nina has not completed every stated administrative-access requirement."},

    # Unit conversion, probability, and ordering.
    {"id": "unit_conversion", "domain": "quantitative", "truth": True,
     "state": "One kilometer equals 1000 meters. The trail is 3.5 kilometers long.",
     "instructions": "Is the trail 3500 meters long?",
     "true": "3.5 kilometers equals 3500 meters.",
     "false": "3.5 kilometers does not equal 3500 meters."},
    {"id": "probability_without_replacement", "domain": "probability", "truth": False,
     "state": "A bag has 3 red balls and 2 blue balls. Two balls are drawn without replacement.",
     "instructions": "Is the probability that both drawn balls are red equal to 1/2?",
     "true": "The probability of two red draws without replacement is 1/2.",
     "false": "The probability of two red draws without replacement is not 1/2."},
    {"id": "sort_order", "domain": "data", "truth": False,
     "state": "The numbers in the sequence are 2, 5, 5, 9, 12.",
     "instructions": "Is the sequence strictly increasing?",
     "true": "Each number is greater than the number immediately before it.",
     "false": "At least one adjacent pair is not strictly increasing."},

    # Language and relational facts where all facts are supplied.
    {"id": "kinship_transitive", "domain": "relational", "truth": True,
     "state": "Ava is Ben's parent. Ben is Cara's parent.",
     "instructions": "Is Ava Cara's grandparent?",
     "true": "Ava is the parent of Cara's parent.",
     "false": "Ava is not the parent of Cara's parent."},
    {"id": "kinship_inverse", "domain": "relational", "truth": False,
     "state": "Ava is Ben's parent. Ben is Cara's parent.",
     "instructions": "Is Cara Ava's parent?",
     "true": "Cara is Ava's parent.",
     "false": "Cara is not Ava's parent."},
    {"id": "contradiction", "domain": "logic", "truth": False,
     "state": "Every submitted expense report must include a receipt. This expense report has no receipt attached.",
     "instructions": "Does the stated information establish that this expense report satisfies every requirement?",
     "true": "The report has the required receipt and satisfies every requirement.",
     "false": "The report lacks a stated required receipt."},
]


def key():
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    value = os.getenv("OPENROUTER_API_KEY", "")
    if value:
        return value
    raise SystemExit("OPENROUTER_API_KEY missing")

def post(url, body, headers, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), round((time.perf_counter() - started) * 1000, 1)

def brier(p, truth): return (p - float(truth)) ** 2

def main():
    rows = []
    for case in CASES:
        criteria = {"true": case["true"], "false": case["false"]}
        question = {"type": "noul", "instructions": case["instructions"], "criteria": criteria}
        jev, jev_ms = post("https://openrouter.ai/api/alpha/decisions", {
            "model": "typesafe/jev-1.13", "state": case["state"], "questions": {case["id"]: question},
        }, {"Authorization": "Bearer " + key(), "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost", "X-Title": "dlgo Jev parity suite"})
        local, local_ms = post(LOCAL_URL, {
            "model": "qwen35-9b", "readout": READOUT, "state": case["state"], "questions": {case["id"]: question},
        }, {"Content-Type": "application/json"})
        jp = jev["answers"][case["id"]]["noul"]
        lp = local["answers"][case["id"]]["noul"]
        truth = case["truth"]
        row = {"id": case["id"], "domain": case["domain"], "truth": truth,
               "jev_p_yes": jp, "jev_correct": (jp >= .5) == truth, "jev_brier": brier(jp, truth), "jev_wall_ms": jev_ms,
               "dlgo_p_yes": lp, "dlgo_correct": (lp >= .5) == truth, "dlgo_brier": brier(lp, truth), "dlgo_wall_ms": local_ms,
               "jev_model": jev.get("model"), "jev_usage": jev.get("usage")}
        rows.append(row)
        print(f"{case['id']:<22} truth={'yes' if truth else 'no ':<3}  jev={jp:.3f} {'OK' if row['jev_correct'] else 'BAD'}  dlgo={lp:.3f} {'OK' if row['dlgo_correct'] else 'BAD'}")
    summary = {}
    for system in ("jev", "dlgo"):
        summary[system] = {"accuracy": sum(r[f"{system}_correct"] for r in rows) / len(rows),
                           "mean_brier": sum(r[f"{system}_brier"] for r in rows) / len(rows),
                           "mean_wall_ms": sum(r[f"{system}_wall_ms"] for r in rows) / len(rows)}
    result = {"suite_version": 1, "cases": rows, "summary": summary}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nSummary:", json.dumps(summary, indent=2))
    print("saved:", OUT)

if __name__ == "__main__": main()
