"""
@Author: hzx
@Date: 2025-04-25
@Version: 1.0
"""

import os
import re

HEADER = """
@Author: hzx
@Date: 2025-04-25
@Version: 1.0
"""

HEADER_PATTERN = re.compile(
    r'^\s*"""'
    r'(?P<header>(\s*@Author:\s*\S+\s+'
    r'\s*@Date:\s*\d{4}-\d{2}-\d{2}\s+'
    r'\s*@Version:\s*\d+\.\d+\s*)+)'
    r'\s*"""',
    re.MULTILINE
)

def update_or_add_header(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if HEADER_PATTERN.search(content):
        new_content = HEADER_PATTERN.sub(f'"""{HEADER}"""', content)
    else:
        new_content = f'"""{HEADER}"""\n\n' + content

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Updated or added header in {file_path}")

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                update_or_add_header(file_path)

if __name__ == "__main__":
    project_directory = "./"
    process_directory(project_directory)