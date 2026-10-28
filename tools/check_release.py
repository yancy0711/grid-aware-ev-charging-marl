"""Check the publication files with only the Python standard library.

By default the directory must contain only release files, plus .git.
With --staged, check the complete Git index, including staged contents.
This is a bounded content check, not a comprehensive secret detector.
"""

import argparse
import ast
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    '.gitignore', 'README.md', 'CITATION.cff', 'environment.yml',
    'env.py', 'rl_charge_extensions.py', 'agents/maddpg.py',
    'agents/single_agent.py', 'train/train_maddpg.py', 'utils/home.py',
    'tools/check_release.py', 'LICENSE',
}
ALLOWED = REQUIRED


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='Check every file in the Git index')
    args = parser.parse_args()
    errors = []
    if args.staged:
        try:
            if Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve() != ROOT:
                raise ValueError('Run in a separately initialized release repository.')
            names = set(git('ls-files', '-z').decode().rstrip('\0').split('\0')) - {''}
            contents = {name: git('show', ':' + name) for name in names & ALLOWED}
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f'FAIL: {exc}', file=sys.stderr)
            return 1
    else:
        names = set()
        contents = {}

        def visit(directory):
            for file in directory.iterdir():
                if file.name == '.git' and file.parent == ROOT:
                    continue
                name = file.relative_to(ROOT).as_posix()
                if file.is_symlink():
                    errors.append(f'Symlink is not allowed: {name}')
                elif file.is_dir():
                    visit(file)
                else:
                    names.add(name)
                    if name in ALLOWED:
                        contents[name] = file.read_bytes()
        visit(ROOT)

    errors.extend(f'Unexpected file: {name}' for name in sorted(names - ALLOWED))
    errors.extend(f'Missing file: {name}' for name in sorted(REQUIRED - names))
    # Composed fragments keep the checker free of the text it rejects.
    forbidden = re.compile(
        r'\b' + 'I' + r'RP\b|independent\s+research\s+project|'
        + 'disser' + r'tation|\bED' + r'SML\b|imper' + 'ial|'
        + r'\bCon' + r'tact\b|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|'
        + r'(?<![A-Za-z0-9_])[A-Z]:[\\/]|/Us' + r'ers/|/ho' + r'me/', re.I
    )
    credentials = re.compile(
        'A' + r'Iza[0-9A-Za-z_-]{35}|gh' + r'[pousr]_[A-Za-z0-9]{20,}|'
        + 'github' + r'_pat_[A-Za-z0-9_]{20,}|AK' + r'IA[A-Z0-9]{16}|'
        + r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'
    )
    for name, raw in sorted(contents.items()):
        if len(raw) > 256_000:
            errors.append(f'File exceeds 256 KB: {name}')
            continue
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            errors.append(f'Not UTF-8 text: {name}')
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if forbidden.search(line) or credentials.search(line):
                errors.append(f'Review project/personal information or credentials: {name}:{lineno}')
        if name.endswith('.py'):
            try:
                ast.parse(text, filename=name)
            except SyntaxError as exc:
                errors.append(f'Syntax error: {name}:{exc.lineno}: {exc.msg}')
        if args.staged:
            mode = git('ls-files', '--stage', '--', name).split(b' ', 1)[0]
            if mode not in {b'100644', b'100755'}:
                errors.append(f'Only regular files are allowed: {name}')
    if errors:
        print('FAIL\n' + '\n'.join(errors), file=sys.stderr)
        return 1
    print(f'PASS: {len(names)} selected text files; source syntax and bounded content checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
