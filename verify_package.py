"""Validate the submission files using only Python's standard library."""
from pathlib import Path
import ast
import sqlite3
import struct

root = Path(__file__).resolve().parent
required = [
    'university_assistant.py', 'README.md', 'short_report.md',
    'architecture.png', 'architecture.mmd', 'university_courses_demo.db'
]
for name in required:
    path = root / name
    assert path.is_file() and path.stat().st_size > 0, f'Missing or empty: {name}'

ast.parse((root / 'university_assistant.py').read_text(encoding='utf-8'))
with sqlite3.connect(root / 'university_courses_demo.db') as connection:
    assert connection.execute('SELECT COUNT(*) FROM courses').fetchone()[0] == 5
    assert connection.execute('SELECT prerequisites FROM courses WHERE code=?', ('ML101',)).fetchone()[0] == 'Python, Statistics, Linear Algebra'

with (root / 'architecture.png').open('rb') as f:
    assert f.read(8) == bytes.fromhex('89504e470d0a1a0a'), 'Invalid PNG signature'

print('PASS: project files exist')
print('PASS: Python syntax')
print('PASS: SQLite database and sample query')
print('PASS: diagram PNG header')
print('NOTE: Groq live calls require installation, network, and your API key.')
print('NOTE: The executed Colab notebook must be exported from your account.')
