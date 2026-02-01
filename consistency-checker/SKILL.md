---
name: consistency-checker
description: Check and fix codebase consistency issues including requirements consolidation, .env/gitignore management, import validation, pipeline data flow verification (output-to-input matching), naming conventions, type consistency, and README updates. Use when setting up/auditing Python projects, before deployment, debugging imports/dependencies, validating pipelines, enforcing standards, or onboarding developers.
---

# Consistency Checker

Automatically detect and fix consistency issues across your codebase to maintain code quality and prevent common errors.

## Overview

This skill provides automated tools to:
1. **Scan** the codebase for consistency issues
2. **Report** detailed findings with severity levels
3. **Fix** issues automatically where safe
4. **Validate** pipeline data flows

## Quick Start

### 1. Check for Issues

Run the consistency checker on your project:

```bash
python scripts/check_consistency.py /path/to/project
```

This generates a detailed report and saves it as `consistency_report.json`.

### 2. Review the Report

The report categorizes issues by:
- 🔴 **High severity**: Security or breaking issues (e.g., .env not in gitignore)
- 🟡 **Medium severity**: Quality issues (e.g., inconsistent naming)
- 🟢 **Low severity**: Nice-to-have improvements

### 3. Apply Fixes

Fix issues automatically:

```bash
# Interactive mode (prompts for confirmation)
python scripts/fix_consistency.py /path/to/project

# Auto mode (applies all safe fixes)
python scripts/fix_consistency.py /path/to/project --auto
```

### 4. Check Pipeline Data Flow

For projects with data pipelines:

```bash
python scripts/check_pipeline.py /path/to/project
```

## What Gets Checked

### Requirements Files
- **Issue**: Multiple `requirements.txt` files across directories
- **Check**: Scans for all `*requirements*.txt` and `*requirement.txt` files
- **Fix**: Consolidates into single root-level `requirements.txt`
- **Backups**: Saves originals to `.consistency_backup/`

### Environment Configuration
- **Issue**: `.env` files not properly managed
- **Checks**:
  - `.env` is in `.gitignore`
  - `.env.example` exists as template
  - Environment variables are documented
- **Fix**: 
  - Adds `.env` to `.gitignore`
  - Creates `.env.example` from existing `.env`

### Git Configuration
- **Issue**: Missing essential patterns in `.gitignore`
- **Checks**: Presence of:
  - `.env` (security)
  - `__pycache__/` (Python cache)
  - `*.pyc` (compiled Python)
  - `node_modules/` (if using Node.js)
- **Fix**: Creates or updates `.gitignore` with essential patterns

### Naming Conventions
- **Issue**: Inconsistent file/folder naming
- **Checks**:
  - Python files use `snake_case`
  - Directories use lowercase with underscores
  - No spaces in names
- **Reports**: Files that don't follow conventions (doesn't auto-fix names to prevent breaking imports)

### Import Statements
- **Issue**: Broken or problematic imports
- **Checks**:
  - Relative imports that might break
  - Missing `__init__.py` files
  - Import statement patterns
- **Reports**: Files with potential import issues for manual review

### Pipeline Data Flow
- **Issue**: Output of one stage doesn't match input of next
- **Checks**:
  - Type hints on functions
  - Return types vs parameter types
  - Data structure compatibility
- **Reports**: Mismatches with suggestions for fixes

See [references/common-issues.md](references/common-issues.md) for detailed solutions.

## Workflow

### Standard Audit Workflow

1. **Initial Scan**
   ```bash
   python scripts/check_consistency.py /path/to/project
   ```
   Review the generated `consistency_report.json` to understand issues.

2. **Critical Fixes First**
   Handle high-severity issues:
   - Ensure `.env` is in `.gitignore`
   - Consolidate requirements files
   - Fix broken imports

3. **Automated Fixes**
   ```bash
   python scripts/fix_consistency.py /path/to/project --auto
   ```
   
4. **Manual Review**
   Review any issues that require manual intervention:
   - Naming convention violations
   - Import structure refactoring
   - Pipeline type mismatches

5. **Validation**
   Re-run the checker to confirm issues are resolved:
   ```bash
   python scripts/check_consistency.py /path/to/project
   ```

### Pipeline-Specific Workflow

For projects with data processing pipelines:

1. **Identify Stages**
   The checker automatically detects files matching:
   - `**/pipeline*.py`
   - `**/stage*.py`
   - `**/*_stage.py`
   - `**/process*.py`

2. **Analyze Data Flow**
   ```bash
   python scripts/check_pipeline.py /path/to/project
   ```

3. **Review Mismatches**
   Check the report for:
   - Type incompatibilities between stages
   - Suggested transformations
   - Missing type hints

4. **Fix Type Issues**
   Add explicit type hints:
   ```python
   def stage1() -> pd.DataFrame:
       """Stage 1 output."""
       return data
   
   def stage2(input_data: pd.DataFrame) -> Dict:
       """Stage 2 expects DataFrame input."""
       return processed
   ```

5. **Add Adapters** (if needed)
   Create adapter functions for type conversions:
   ```python
   def adapt_stage1_to_stage2(output: List[Dict]) -> pd.DataFrame:
       """Convert stage1 output to stage2 input format."""
       return pd.DataFrame(output)
   ```

## Advanced Usage

### Custom Checks

Extend the checker for project-specific rules:

```python
from scripts.check_consistency import ConsistencyChecker

class MyChecker(ConsistencyChecker):
    def check_custom_rule(self):
        """Add your custom consistency check."""
        # Your logic here
        if issue_found:
            self.issues["custom"].append({
                "severity": "medium",
                "message": "Custom issue found",
                "files": ["path/to/file"]
            })

checker = MyChecker("/path/to/project")
report = checker.check_all()
```

### Filtering Reports

Focus on specific categories:

```python
import json

with open('consistency_report.json') as f:
    report = json.load(f)

# Show only high-severity issues
high_severity = [
    issue for issues in report['issues'].values()
    for issue in issues
    if issue.get('severity') == 'high'
]
```

### CI/CD Integration

Add to your CI pipeline:

```yaml
# .github/workflows/consistency-check.yml
name: Consistency Check

on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check Consistency
        run: |
          python scripts/check_consistency.py .
          if grep -q '"severity": "high"' consistency_report.json; then
            echo "High severity issues found!"
            exit 1
          fi
```

## Output Files

All scripts generate reports saved in the project root:

- `consistency_report.json`: Full detailed report from checker
- `pipeline_report.json`: Pipeline-specific analysis
- `.consistency_backup/`: Backup of files before automated changes

## Best Practices

1. **Run Before Commits**: Check consistency before committing
2. **Review Auto-Fixes**: Always review changes made in `--auto` mode
3. **Keep Backups**: The tool creates backups, but verify critical files
4. **Iterative Fixing**: Fix high-severity issues first, then medium, then low
5. **Document Decisions**: If you intentionally violate a rule, document why

## Limitations

- **Naming**: Won't auto-rename files (risks breaking imports)
- **Imports**: Can't fix all import issues automatically (complex dependencies)
- **Pipelines**: Requires type hints for accurate analysis
- **README**: Only checks existence, not content quality

For manual intervention guidance, see [references/common-issues.md](references/common-issues.md).

## Troubleshooting

### "No pipeline stages detected"
- Ensure files follow naming patterns: `pipeline*.py`, `*_stage.py`, etc.
- Pipeline files should contain processing functions

### "Import analysis failed"
- Check for syntax errors in Python files
- Ensure files are valid Python (`.py` extension)

### "Permission denied" during fixes
- Run with appropriate permissions
- Check file/directory ownership

### False positives
- Review the report context
- Add exceptions in custom checker if needed

## Reference Materials

- **[common-issues.md](references/common-issues.md)**: Detailed guide for each issue type with multiple solution approaches
