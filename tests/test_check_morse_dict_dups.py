import ast
import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _check_morse_dict_dups_py(file_path):
    if not os.path.exists(file_path):
        raise AssertionError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'MORSE_DICT':
                    if isinstance(node.value, ast.Dict):
                        keys = []
                        dups = []
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant):
                                k = key.value
                                if k in keys:
                                    dups.append(k)
                                keys.append(k)
                        
                        assert not dups, f"Duplicate keys found in MORSE_DICT in {file_path}: {dups}"
                        return
    
    raise AssertionError(f"MORSE_DICT not found in {file_path}.")


def _check_morse_dict_dups_js(file_path):
    if not os.path.exists(file_path):
        raise AssertionError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    match = re.search(r'export const MORSE_DICT = \{(.*?)\};', content, re.DOTALL)
    assert match, f"MORSE_DICT not found in {file_path}"
    
    dict_content = match.group(1)
    keys = re.findall(r"['\"](.*?)['\"]\s*:", dict_content)
    seen = set()
    dups = []
    for k in keys:
        if k in seen:
            dups.append(k)
        seen.add(k)
    
    assert not dups, f"Duplicate keys found in {file_path}: {dups}"


def test_check_morse_dict_dups():
    _check_morse_dict_dups_py('data_gen.py')
    _check_morse_dict_dups_js('demo/data_gen.js')
