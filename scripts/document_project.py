import os
import pathlib
from typing import List, Dict, Optional
import markdown
from dataclasses import dataclass

@dataclass
class FileInfo:
    path: str
    is_dir: bool
    size: int
    content: Optional[str] = None
    description: str = ""

def get_project_structure(root_dir: str, ignore_dirs: List[str] = None) -> Dict[str, FileInfo]:
    """
    Recursively get the project structure and file information.
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.pytest_cache', '.venv', 'venv', '.gradio', '.idea']

    project_structure = {}
    root_path = pathlib.Path(root_dir)
    
    # First add all directories to maintain hierarchy
    for dirpath, dirnames, _ in os.walk(root_dir):
        # Remove ignored directories from dirnames to prevent os.walk from traversing them
        dirnames[:] = [d for d in dirnames if not any(ignore in d for ignore in ignore_dirs)]
        
        relative_path = str(pathlib.Path(dirpath).relative_to(root_dir))
        if relative_path == '.':
            relative_path = ''
            
        # Add current directory if it's not the root
        if relative_path and relative_path not in project_structure:
            project_structure[relative_path] = FileInfo(
                path=relative_path,
                is_dir=True,
                size=0
            )
            
        # Add all subdirectories
        for dirname in dirnames:
            dir_rel_path = os.path.join(relative_path, dirname) if relative_path else dirname
            project_structure[dir_rel_path] = FileInfo(
                path=dir_rel_path,
                is_dir=True,
                size=0
            )
    
    # Then process all files
    for item in root_path.rglob('*'):
        if any(ignore in str(item) for ignore in ignore_dirs):
            continue
            
        if item.is_file():
            relative_path = str(item.relative_to(root_dir))
            size = item.stat().st_size
            
            file_info = FileInfo(
                path=relative_path,
                is_dir=False,
                size=size
            )
            
            # Read content of relevant files
            if should_include_file(relative_path):
                try:
                    with open(item, 'r', encoding='utf-8') as f:
                        file_info.content = f.read()
                except (UnicodeDecodeError, PermissionError):
                    file_info.content = "[Binary or unreadable file]"
            
            project_structure[relative_path] = file_info

    return project_structure

def should_include_file(file_path: str) -> bool:
    """Determine if a file should have its content included in the documentation."""
    include_extensions = {'.py', '.md', '.txt', '.yaml', '.yml', '.json', '.sh'}
    exclude_files = {'__init__.py'}

    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)

    if file_name in exclude_files:
        return False
    return ext.lower() in include_extensions

def generate_directory_tree(project_structure: Dict[str, FileInfo]) -> str:
    """Generate a proper directory tree structure."""
    tree_lines = []
    dirs = []
    files = []
    
    # Separate directories and files
    for rel_path, info in project_structure.items():
        if info.is_dir:
            dirs.append((rel_path, info))
        else:
            files.append((rel_path, info))
    
    # Sort directories and files
    dirs.sort()
    files.sort()
    
    # Process directories with proper indentation
    for rel_path, info in dirs:
        parts = rel_path.split(os.sep)
        for i in range(1, len(parts) + 1):
            current_path = os.path.join(*parts[:i])
            if current_path not in project_structure:
                continue
                
            indent = '    ' * (i - 1)
            if i == len(parts):
                tree_lines.append(f"{indent}└── {parts[-1]}/")
            elif i == 1 and current_path not in [d[0] for d in dirs if d[0] != rel_path and d[0].startswith(parts[0])]:
                tree_lines.append(f"{indent}├── {parts[i-1]}/")
    
    # Process files with proper indentation
    for rel_path, info in files:
        parts = rel_path.split(os.sep)
        if len(parts) > 1:
            # Find the last directory that exists in our structure
            for i in range(len(parts)-1, 0, -1):
                parent_path = os.path.join(*parts[:i])
                if parent_path in project_structure:
                    indent = '    ' * i
                    tree_lines.append(f"{indent}└── {parts[-1]} ({info.size} bytes)")
                    break
        else:
            tree_lines.append(f"├── {rel_path} ({info.size} bytes)")
    
    return '\n'.join([
        ".",
        *tree_lines
    ])

def generate_markdown_docs(project_structure: Dict[str, FileInfo], output_file: str = "PROJECT_DOCS.md"):
    """Generate markdown documentation from the project structure."""
    with open(output_file, 'w', encoding='utf-8') as f:
        # Project title
        project_name = os.path.basename(os.path.abspath('.'))
        f.write(f"# {project_name} Project Documentation\n\n")

        # Table of Contents
        f.write("## Table of Contents\n")
        f.write("- [Project Structure](#project-structure)\n")
        f.write("- [Key Files](#key-files)\n")

        # Project Structure
        f.write("\n## Project Structure\n\n```")
        f.write("\n" + generate_directory_tree(project_structure))
        f.write("\n```\n\n")

        # Key Files Section
        f.write("## Key Files\n\n")

        # Sort items: directories first, then files, both alphabetically
        sorted_items = sorted(project_structure.items(), 
                            key=lambda x: (not x[1].is_dir, x[0]))

        for rel_path, info in sorted_items:
            if not info.is_dir and info.content is not None:
                f.write(f"### `{rel_path}`\n\n")
                f.write(f"```{get_file_extension(rel_path)}\n")
                f.write(info.content[:2000])  # First 2000 chars to avoid huge output
                if len(info.content) > 2000:
                    f.write("\n... [content truncated]")
                f.write("\n```\n\n")
                f.write("---\n\n")

def get_file_extension(file_path: str) -> str:
    """Get the file extension for syntax highlighting."""
    _, ext = os.path.splitext(file_path)
    return ext[1:] if ext else 'text'

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(project_root, "PROJECT_DOCS.md")

    print(f"Documenting project at: {project_root}")
    project_structure = get_project_structure(project_root)
    generate_markdown_docs(project_structure, output_file)
    print(f"Documentation generated at: {output_file}")

if __name__ == "__main__":
    main()
