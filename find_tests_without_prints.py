import ast
from pathlib import Path


def file_has_print(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == 'print':
                return True
    return False


def main():
    tests_dir = Path('tests')
    missing = []
    for path in sorted(tests_dir.glob('test_*.py')):
        if not file_has_print(path):
            missing.append(path)
    for path in missing:
        print(path)
    print(f"Total: {len(missing)} tests without prints.")


if __name__ == '__main__':
    main()
