import ast
import os

def check_morse_dict_dups(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

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
                        
                        if dups:
                            print(f"Duplicate keys found in MORSE_DICT in {file_path}:")
                            for d in dups:
                                print(f"  - {repr(d)}")
                            return True
                        else:
                            print(f"No duplicate keys found in MORSE_DICT in {file_path}.")
                            return False
    
    print(f"MORSE_DICT not found in {file_path}.")
    return False

if __name__ == "__main__":
    has_dups_py = check_morse_dict_dups('data_gen.py')
    # JS version check (simple regex based for speed)
    print("\nChecking demo/data_gen.js (simple check)...")
    with open('demo/data_gen.js', 'r') as f:
        content = f.read()
        import re
        # Find MORSE_DICT content
        match = re.search(r'export const MORSE_DICT = \{(.*?)\};', content, re.DOTALL)
        if match:
            dict_content = match.group(1)
            # Find all keys
            keys = re.findall(r"['\"](.*?)['\"]\s*:", dict_content)
            seen = set()
            dups = []
            for k in keys:
                if k in seen:
                    dups.append(k)
                seen.add(k)
            if dups:
                print(f"Duplicate keys found in demo/data_gen.js:")
                for d in dups:
                    print(f"  - {repr(d)}")
            else:
                print("No duplicate keys found in demo/data_gen.js.")