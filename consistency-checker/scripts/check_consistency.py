#!/usr/bin/env python3
"""
Consistency Checker - Main Script
Scans a codebase for various consistency issues and generates a detailed report.
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class ConsistencyChecker:
    """Main class for checking codebase consistency."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.issues = defaultdict(list)
        self.suggestions = defaultdict(list)
        
    def check_all(self) -> Dict:
        """Run all consistency checks."""
        print(f"🔍 Scanning {self.root_dir}...")
        
        self.check_requirements_files()
        self.check_env_files()
        self.check_gitignore()
        self.check_naming_conventions()
        self.check_python_imports()
        
        return self.generate_report()
    
    def check_requirements_files(self):
        """Check for multiple requirements.txt files."""
        requirements_files = list(self.root_dir.rglob("*requirements*.txt"))
        requirement_files = list(self.root_dir.rglob("*requirement.txt"))
        all_req_files = requirements_files + requirement_files
        
        if len(all_req_files) > 1:
            paths = [str(f.relative_to(self.root_dir)) for f in all_req_files]
            self.issues["requirements"].append({
                "severity": "high",
                "message": f"Found {len(all_req_files)} requirements files",
                "files": paths
            })
            self.suggestions["requirements"].append(
                "Consolidate all requirements into a single requirements.txt at the root"
            )
        elif len(all_req_files) == 0:
            self.issues["requirements"].append({
                "severity": "medium",
                "message": "No requirements.txt file found",
                "files": []
            })
            self.suggestions["requirements"].append(
                "Create a requirements.txt file at the project root"
            )
    
    def check_env_files(self):
        """Check for .env files and their configuration."""
        env_files = list(self.root_dir.rglob(".env*"))
        example_env_files = list(self.root_dir.rglob(".env.example"))
        
        if not env_files and not example_env_files:
            self.issues["env"].append({
                "severity": "low",
                "message": "No .env or .env.example files found",
                "files": []
            })
            self.suggestions["env"].append(
                "Consider creating a .env.example file with placeholder values"
            )
        
        # Check if .env files are in .gitignore
        gitignore_path = self.root_dir / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                gitignore_content = f.read()
            
            if ".env" not in gitignore_content:
                self.issues["env"].append({
                    "severity": "high",
                    "message": ".env is not in .gitignore",
                    "files": [".gitignore"]
                })
                self.suggestions["env"].append(
                    "Add .env to .gitignore to prevent sensitive data from being committed"
                )
    
    def check_gitignore(self):
        """Check if .gitignore exists and contains essential entries."""
        gitignore_path = self.root_dir / ".gitignore"
        
        if not gitignore_path.exists():
            self.issues["gitignore"].append({
                "severity": "medium",
                "message": "No .gitignore file found",
                "files": []
            })
            self.suggestions["gitignore"].append(
                "Create a .gitignore file with common patterns (.env, __pycache__, etc.)"
            )
        else:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                gitignore_content = f.read()
            
            essential_patterns = {
                ".env": ".env file should be ignored",
                "__pycache__": "Python cache should be ignored",
                "*.pyc": "Compiled Python files should be ignored",
                "node_modules": "Node modules should be ignored (if using Node.js)"
            }
            
            for pattern, reason in essential_patterns.items():
                if pattern not in gitignore_content:
                    self.issues["gitignore"].append({
                        "severity": "medium",
                        "message": f"Missing pattern: {pattern}",
                        "files": [".gitignore"]
                    })
                    self.suggestions["gitignore"].append(f"Add '{pattern}' to .gitignore - {reason}")
    
    def check_naming_conventions(self):
        """Check folder and file naming consistency."""
        inconsistent_names = []
        
        # Python convention: lowercase with underscores
        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.suffix == ".py":
                name = path.stem
                # Check if name follows snake_case
                if not re.match(r'^[a-z0-9_]+$', name) and name != "__init__":
                    inconsistent_names.append({
                        "file": str(path.relative_to(self.root_dir)),
                        "reason": "Python files should use snake_case"
                    })
            
            if path.is_dir() and path.parent != self.root_dir.parent:
                name = path.name
                # Skip hidden directories and common exceptions
                if name.startswith('.') or name in ['node_modules', '__pycache__', 'venv', 'env']:
                    continue
                
                # Check for mixed case or spaces
                if ' ' in name or (name != name.lower() and '_' not in name):
                    inconsistent_names.append({
                        "file": str(path.relative_to(self.root_dir)),
                        "reason": "Directories should use lowercase with underscores"
                    })
        
        if inconsistent_names:
            self.issues["naming"].append({
                "severity": "medium",
                "message": f"Found {len(inconsistent_names)} naming inconsistencies",
                "files": inconsistent_names
            })
            self.suggestions["naming"].append(
                "Use consistent naming: snake_case for Python files/dirs, kebab-case or PascalCase for others"
            )
    
    def check_python_imports(self):
        """Check for potential import issues in Python files."""
        import_issues = []
        
        for py_file in self.root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all imports
                import_lines = re.findall(r'^(?:from|import)\s+[\w.]+', content, re.MULTILINE)
                
                # Check for relative imports that might break
                relative_imports = re.findall(r'^from\s+\.+[\w.]*\s+import', content, re.MULTILINE)
                if relative_imports:
                    import_issues.append({
                        "file": str(py_file.relative_to(self.root_dir)),
                        "issue": "Uses relative imports - verify module structure",
                        "count": len(relative_imports)
                    })
                
            except Exception as e:
                import_issues.append({
                    "file": str(py_file.relative_to(self.root_dir)),
                    "issue": f"Could not analyze: {str(e)}"
                })
        
        if import_issues:
            self.issues["imports"].append({
                "severity": "medium",
                "message": f"Found {len(import_issues)} files with import concerns",
                "files": import_issues
            })
            self.suggestions["imports"].append(
                "Review relative imports and ensure __init__.py files are present in all packages"
            )
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive report of all issues."""
        report = {
            "summary": {
                "total_issues": sum(len(v) for v in self.issues.values()),
                "categories": list(self.issues.keys())
            },
            "issues": dict(self.issues),
            "suggestions": dict(self.suggestions)
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print a formatted report to console."""
        print("\n" + "="*60)
        print("📊 CONSISTENCY CHECK REPORT")
        print("="*60)
        
        print(f"\n📈 Summary:")
        print(f"   Total Issues: {report['summary']['total_issues']}")
        print(f"   Categories: {', '.join(report['summary']['categories'])}")
        
        for category, issues in report['issues'].items():
            print(f"\n🔍 {category.upper()}")
            print("-" * 60)
            for issue in issues:
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                severity = issue.get("severity", "medium")
                print(f"{severity_emoji[severity]} {issue['message']}")
                
                if isinstance(issue.get('files'), list) and issue['files']:
                    if isinstance(issue['files'][0], dict):
                        for file_info in issue['files'][:5]:  # Show first 5
                            print(f"   - {file_info.get('file', file_info)}")
                            if 'reason' in file_info:
                                print(f"     Reason: {file_info['reason']}")
                    else:
                        for file in issue['files'][:5]:  # Show first 5
                            print(f"   - {file}")
                    
                    if len(issue['files']) > 5:
                        print(f"   ... and {len(issue['files']) - 5} more")
        
        if report['suggestions']:
            print(f"\n💡 SUGGESTIONS")
            print("-" * 60)
            for category, suggestions in report['suggestions'].items():
                print(f"\n{category.upper()}:")
                for suggestion in suggestions:
                    print(f"  • {suggestion}")
        
        print("\n" + "="*60)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python check_consistency.py <directory>")
        print("Example: python check_consistency.py /path/to/project")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    
    if not os.path.isdir(root_dir):
        print(f"❌ Error: {root_dir} is not a valid directory")
        sys.exit(1)
    
    checker = ConsistencyChecker(root_dir)
    report = checker.check_all()
    checker.print_report(report)
    
    # Save report to JSON
    report_path = Path(root_dir) / "consistency_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
