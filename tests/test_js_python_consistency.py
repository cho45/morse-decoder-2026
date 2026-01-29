import re
import config
from data_gen import MORSE_DICT

def extract_js_morse_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract MORSE_DICT object content
    match = re.search(r"export const MORSE_DICT = \{(.*?)\};", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find MORSE_DICT in {file_path}")
    
    dict_content = match.group(1)
    
    # Robust parsing for JS object: handles escaped quotes and various delimiters
    # Group 1-2: key (quoted), Group 3: key (raw), Group 4-5: value (quoted)
    pattern = r'(?:"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'|([a-zA-Z0-9<>]+))\s*:\s*(?:"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\')'
    pairs = re.findall(pattern, dict_content)
    
    result = {}
    for k_dq, k_sq, k_raw, v_dq, v_sq in pairs:
        key = k_dq or k_sq or k_raw
        val = v_dq or v_sq
        # Unescape JS strings
        key = key.replace('\\"', '"').replace("\\'", "'")
        val = val.replace('\\"', '"').replace("\\'", "'")
        result[key] = val
    return result

def extract_js_chars(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract CHARS array content
    match = re.search(r"export const CHARS = \[(.*?)\];", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find CHARS in {file_path}")
    
    array_content = match.group(1)
    
    # Robust parsing for JS string array: handles escaped quotes and various delimiters
    # Matches "..." or '...' while respecting escaped characters
    items = re.findall(r'"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'', array_content)
    
    processed_items = []
    for dq, sq in items:
        item = dq if dq else sq
        # Unescape JS string
        item = item.replace('\\"', '"').replace("\\'", "'")
        processed_items.append(item)
        
    return processed_items

def test_morse_dict_consistency():
    """Verify MORSE_DICT consistency between Python and JS."""
    js_morse = extract_js_morse_dict("demo/data_gen.js")
    py_morse = MORSE_DICT
    
    # 1. Check keys
    js_keys = set(js_morse.keys())
    py_keys = set(py_morse.keys())
    
    diff_keys_py = py_keys - js_keys
    diff_keys_js = js_keys - py_keys
    
    # 2. Check values for common keys
    mismatched_values = {}
    for key in js_keys & py_keys:
        if js_morse[key] != py_morse[key]:
            mismatched_values[key] = {"js": js_morse[key], "py": py_morse[key]}
            
    error_msg = []
    if diff_keys_py:
        error_msg.append(f"Keys in Python but not in JS: {diff_keys_py}")
    if diff_keys_js:
        error_msg.append(f"Keys in JS but not in Python: {diff_keys_js}")
    if mismatched_values:
        error_msg.append(f"Mismatched values: {mismatched_values}")
        
    assert not error_msg, "\n".join(error_msg)

def test_chars_consistency():
    """Verify CHARS (vocabulary) consistency between Python and JS, including order."""
    js_chars = extract_js_chars("demo/inference.js")
    py_chars = config.CHARS
    
    # Check length
    if len(js_chars) != len(py_chars):
        print(f"Length mismatch: JS={len(js_chars)}, Python={len(py_chars)}")
    
    # Check content and order
    mismatches = []
    max_len = max(len(js_chars), len(py_chars))
    for i in range(max_len):
        js_val = js_chars[i] if i < len(js_chars) else None
        py_val = py_chars[i] if i < len(py_chars) else None
        if js_val != py_val:
            mismatches.append(f"Index {i}: JS='{js_val}', Python='{py_val}'")
            
    assert not mismatches, f"CHARS mismatch found:\n" + "\n".join(mismatches)

if __name__ == "__main__":
    # For manual debugging
    try:
        print("Testing Morse Dict consistency...")
        test_morse_dict_consistency()
        print("OK")
    except AssertionError as e:
        print(f"FAILED:\n{e}")
        
    try:
        print("\nTesting CHARS consistency...")
        test_chars_consistency()
        print("OK")
    except AssertionError as e:
        print(f"FAILED:\n{e}")