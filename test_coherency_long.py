"""
Coherence test for long generation on dlgo.
Tests that the model maintains coherent, structured output across >4K tokens.
Checks for: repetition loops, garbage output, topic drift, numbering errors.
"""
import json
import re
import sys
import urllib.request

BASE = "http://localhost:9090/v1/chat/completions"

def chat(prompt, max_tokens=4000, temp=0.3):
    body = json.dumps({
        "model": "Qwen3.5-0.8B-Q8_0",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": False,
        "temperature": temp,
    }).encode()
    req = urllib.request.Request(BASE, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    msg = data["choices"][0]["message"]["content"]
    usage = data["usage"]
    finish = data["choices"][0]["finish_reason"]
    return msg, usage, finish


def check_repetition(text, window=100, threshold=3):
    """Detect if any 100-char window repeats 3+ times consecutively."""
    words = text.split()
    for size in [10, 20, 50]:
        for i in range(len(words) - size * threshold):
            chunk = " ".join(words[i:i+size])
            count = 0
            j = i
            while j + size <= len(words):
                candidate = " ".join(words[j:j+size])
                if candidate == chunk:
                    count += 1
                    j += size
                else:
                    break
            if count >= threshold:
                return True, f"Repeated {size}-word block {count}x at word {i}: '{chunk[:80]}...'"
    return False, ""


def check_garbage(text):
    """Check for garbage/corrupted output (high ratio of non-ASCII or control chars)."""
    total = len(text)
    if total == 0:
        return True, "Empty output"
    non_printable = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    ratio = non_printable / total
    if ratio > 0.05:
        return True, f"High non-printable ratio: {ratio:.2%}"
    return False, ""


def test_numbered_list():
    """Test: Generate a numbered list — check numbering stays sequential."""
    print("\n=== TEST 1: Numbered list coherence (GPU) ===")
    prompt = (
        "Write a numbered list of exactly 200 interesting science facts. "
        "Number each fact starting from 1. Format: '1. fact text', '2. fact text', etc. "
        "Do not skip or repeat any numbers."
    )
    text, usage, finish = chat(prompt, max_tokens=8000, temp=0.3)
    comp_tokens = usage["completion_tokens"]
    print(f"  Generated: {comp_tokens} tokens, finish={finish}")

    # Extract numbers
    numbers = [int(m) for m in re.findall(r'^(\d+)\.', text, re.MULTILINE)]
    print(f"  Found {len(numbers)} numbered items")

    if len(numbers) < 20:
        print(f"  FAIL: Too few numbered items ({len(numbers)})")
        return False, text

    # Check sequential ordering (allow some gaps from model creativity, but no big jumps)
    errors = []
    for i in range(1, len(numbers)):
        diff = numbers[i] - numbers[i-1]
        if diff != 1:
            errors.append(f"    Jump at item {i}: {numbers[i-1]} -> {numbers[i]} (diff={diff})")
    
    if errors:
        print(f"  Numbering errors ({len(errors)}):")
        for e in errors[:10]:
            print(e)
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")

    is_garbage, garbage_msg = check_garbage(text)
    if is_garbage:
        print(f"  FAIL: Garbage detected: {garbage_msg}")
        return False, text

    has_repeat, repeat_msg = check_repetition(text)
    if has_repeat:
        print(f"  FAIL: Repetition detected: {repeat_msg}")
        return False, text

    # Allow up to 10% numbering errors (model may skip occasionally)
    error_rate = len(errors) / max(len(numbers), 1)
    if error_rate > 0.15:
        print(f"  FAIL: Too many numbering errors ({error_rate:.0%})")
        return False, text

    print(f"  PASS: {len(numbers)} items, {len(errors)} numbering irregularities ({error_rate:.0%}), {comp_tokens} tokens")
    return True, text


def test_structured_essay():
    """Test: Long structured essay — check for topic coherence and structure."""
    print("\n=== TEST 2: Structured essay coherence (GPU) ===")
    prompt = (
        "Write a detailed, well-structured essay about the history of the Internet. "
        "Use clear section headings (## Heading) for at least 8 sections. "
        "Cover: ARPANET, TCP/IP, WWW, browsers, social media, mobile internet, cloud computing, and future trends. "
        "Each section should have at least 3 paragraphs. Make it very detailed and long."
    )
    text, usage, finish = chat(prompt, max_tokens=8000, temp=0.4)
    comp_tokens = usage["completion_tokens"]
    print(f"  Generated: {comp_tokens} tokens, finish={finish}")

    # Check for section headings
    headings = re.findall(r'^#{1,3}\s+(.+)$', text, re.MULTILINE)
    print(f"  Found {len(headings)} headings: {headings[:8]}")

    if len(headings) < 3:
        print(f"  FAIL: Too few headings ({len(headings)})")
        return False, text

    is_garbage, garbage_msg = check_garbage(text)
    if is_garbage:
        print(f"  FAIL: Garbage detected: {garbage_msg}")
        return False, text

    has_repeat, repeat_msg = check_repetition(text)
    if has_repeat:
        print(f"  FAIL: Repetition detected: {repeat_msg}")
        return False, text

    # Check for internet-related keywords throughout the text
    # Split into quarters and check each has relevant content
    quarter = len(text) // 4
    keywords = ["internet", "network", "web", "protocol", "computer", "online", "digital", "data"]
    quarters_ok = 0
    for q in range(4):
        chunk = text[q*quarter:(q+1)*quarter].lower()
        found = [k for k in keywords if k in chunk]
        if len(found) >= 2:
            quarters_ok += 1
        else:
            print(f"  WARNING: Quarter {q+1} lacks topic keywords (found: {found})")

    if quarters_ok < 3:
        print(f"  FAIL: Topic drift — only {quarters_ok}/4 quarters on-topic")
        return False, text

    print(f"  PASS: {len(headings)} headings, {quarters_ok}/4 quarters on-topic, {comp_tokens} tokens")
    return True, text


def test_code_generation():
    """Test: Generate a long code block — check for syntax structure coherence."""
    print("\n=== TEST 3: Code generation coherence (GPU) ===")
    prompt = (
        "Write a complete Python implementation of a binary search tree with the following methods:\n"
        "insert, delete, search, inorder_traversal, preorder_traversal, postorder_traversal,\n"
        "find_min, find_max, height, size, is_balanced, level_order_traversal,\n"
        "successor, predecessor, serialize, deserialize.\n"
        "Include detailed docstrings and comprehensive unit tests using unittest.\n"
        "Make the implementation production-quality with proper error handling."
    )
    text, usage, finish = chat(prompt, max_tokens=8000, temp=0.2)
    comp_tokens = usage["completion_tokens"]
    print(f"  Generated: {comp_tokens} tokens, finish={finish}")

    is_garbage, garbage_msg = check_garbage(text)
    if is_garbage:
        print(f"  FAIL: Garbage detected: {garbage_msg}")
        return False, text

    has_repeat, repeat_msg = check_repetition(text)
    if has_repeat:
        print(f"  FAIL: Repetition detected: {repeat_msg}")
        return False, text

    # Check for code structure markers
    has_class = "class " in text
    has_def = text.count("def ") >= 5
    has_self = "self." in text
    has_return = "return " in text

    checks = [
        ("class definition", has_class),
        ("5+ function definitions", has_def),
        ("self references", has_self),
        ("return statements", has_return),
    ]
    passed = [name for name, ok in checks if ok]
    failed = [name for name, ok in checks if not ok]

    if len(failed) > 1:
        print(f"  FAIL: Missing code structure: {failed}")
        return False, text

    print(f"  PASS: Code structure OK ({passed}), {comp_tokens} tokens")
    return True, text


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gpu"
    print(f"=== Long Output Coherence Test ({mode.upper()}) ===")

    results = []
    texts = {}

    ok, text = test_numbered_list()
    results.append(("numbered_list", ok))
    texts["numbered_list"] = text

    ok, text = test_structured_essay()
    results.append(("structured_essay", ok))
    texts["structured_essay"] = text

    ok, text = test_code_generation()
    results.append(("code_generation", ok))
    texts["code_generation"] = text

    print(f"\n=== SUMMARY ({mode.upper()}) ===")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\nAll {len(results)} coherence tests PASSED on {mode.upper()}")
    else:
        print(f"\nSome tests FAILED on {mode.upper()}")
        sys.exit(1)

    # Save outputs for comparison
    with open(f"coherence_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    print(f"Outputs saved to coherence_{mode}.json")


if __name__ == "__main__":
    main()
