# Common Consistency Issues and Solutions

This reference document provides detailed guidance on identifying and fixing common consistency issues in Python codebases.

## Requirements Files

### Issue: Multiple requirements.txt files
**Description**: Having multiple requirements files across subdirectories leads to dependency conflicts and maintenance overhead.

**Detection**:
- Found in: `check_consistency.py::check_requirements_files()`
- Pattern: `*requirements*.txt` or `*requirement.txt`

**Solutions**:
1. **Consolidate to single file** (Recommended)
   ```bash
   # Merge all requirements
   cat **/requirements.txt | sort | uniq > requirements.txt
   ```

2. **Use environment-specific files**
   ```
   requirements.txt          # Core dependencies
   requirements-dev.txt      # Development only
   requirements-prod.txt     # Production only
   ```

3. **Use setup.py or pyproject.toml**
   ```python
   # setup.py
   install_requires=[
       'package1>=1.0.0',
       'package2>=2.0.0',
   ]
   ```

---

## Environment Variables

### Issue: .env not in .gitignore
**Description**: Committing .env files exposes sensitive credentials.

**Detection**:
- Found in: `check_consistency.py::check_env_files()`
- Check: `.env` pattern in `.gitignore`

**Solutions**:
1. **Add to .gitignore immediately**
   ```bash
   echo ".env" >> .gitignore
   git rm --cached .env  # Remove from git if already committed
   ```

2. **Create .env.example**
   ```bash
   # .env.example
   DATABASE_URL=postgresql://user:pass@localhost/db
   API_KEY=your_api_key_here
   SECRET_KEY=your_secret_key_here
   ```

3. **Use environment-specific files**
   ```
   .env.local              # Local development (gitignored)
   .env.example            # Template (committed)
   .env.production         # Production (never committed)
   ```

### Issue: Missing .env.example
**Description**: New developers don't know what environment variables are needed.

**Solutions**:
1. **Create from existing .env**
   ```python
   # Auto-generate
   with open('.env') as f:
       lines = [line.split('=')[0] + '=\n' for line in f if '=' in line]
   with open('.env.example', 'w') as f:
       f.writelines(lines)
   ```

2. **Document in README**
   ```markdown
   ## Environment Setup
   1. Copy `.env.example` to `.env`
   2. Fill in your actual values
   ```

---

## Import Issues

### Issue: Broken relative imports
**Description**: Relative imports fail when module structure changes or scripts run from different locations.

**Detection**:
- Found in: `check_consistency.py::check_python_imports()`
- Pattern: `from . import` or `from .. import`

**Solutions**:
1. **Use absolute imports** (Recommended)
   ```python
   # Bad
   from .utils import helper
   
   # Good
   from myproject.utils import helper
   ```

2. **Add __init__.py files**
   ```
   myproject/
   ├── __init__.py          # Makes it a package
   ├── module1/
   │   ├── __init__.py
   │   └── code.py
   └── module2/
       ├── __init__.py
       └── code.py
   ```

3. **Fix PYTHONPATH**
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```

### Issue: Missing imports
**Description**: Code references modules that aren't imported.

**Solutions**:
1. **Use tools to detect**
   ```bash
   pip install pyflakes
   pyflakes **.py
   ```

2. **Auto-fix with isort and autoflake**
   ```bash
   pip install isort autoflake
   isort --profile black .
   autoflake --remove-all-unused-imports -i **.py
   ```

---

## Naming Conventions

### Issue: Inconsistent file/folder names
**Description**: Mixed naming conventions make the codebase harder to navigate.

**Detection**:
- Found in: `check_consistency.py::check_naming_conventions()`
- Checks: snake_case for Python, consistency across types

**Solutions**:
1. **Python files: snake_case**
   ```
   ❌ myFile.py, My-File.py
   ✅ my_file.py
   ```

2. **Python modules/packages: lowercase with underscores**
   ```
   ❌ MyPackage/, my-package/
   ✅ my_package/
   ```

3. **Classes: PascalCase**
   ```python
   ❌ class my_class:
   ✅ class MyClass:
   ```

4. **Constants: UPPER_SNAKE_CASE**
   ```python
   ❌ max_retries = 3
   ✅ MAX_RETRIES = 3
   ```

---

## Pipeline Data Flow

### Issue: Type mismatches between pipeline stages
**Description**: Output of one stage doesn't match expected input of the next.

**Detection**:
- Found in: `check_pipeline.py::check_data_flow()`
- Analyzes type hints and function signatures

**Solutions**:
1. **Add explicit type hints**
   ```python
   def stage1() -> pd.DataFrame:
       return data
   
   def stage2(input_data: pd.DataFrame) -> Dict:
       return processed
   ```

2. **Use adapters/transformers**
   ```python
   class DataAdapter:
       @staticmethod
       def stage1_to_stage2(output: List) -> pd.DataFrame:
           return pd.DataFrame(output)
   
   # Usage
   stage1_output = stage1()
   stage2_input = DataAdapter.stage1_to_stage2(stage1_output)
   stage2(stage2_input)
   ```

3. **Validate at runtime**
   ```python
   from typing import get_type_hints
   
   def validate_pipeline_input(func, data):
       hints = get_type_hints(func)
       expected = hints.get('input_data')
       if not isinstance(data, expected):
           raise TypeError(f"Expected {expected}, got {type(data)}")
   ```

---

## Git Configuration

### Issue: Essential patterns missing from .gitignore
**Description**: Temporary files, caches, and sensitive data get committed.

**Essential Patterns**:
```gitignore
# Environment
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Logs
*.log
logs/

# Database
*.db
*.sqlite
*.sqlite3

# Node (if applicable)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
```

**Solutions**:
1. **Use templates**
   ```bash
   curl https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore > .gitignore
   ```

2. **Check what's ignored**
   ```bash
   git check-ignore -v *
   ```

---

## README and Documentation

### Issue: Outdated README.md
**Description**: README doesn't reflect current project state.

**Essential Sections**:
```markdown
# Project Name

## Description
What this project does and why it exists

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Copy `.env.example` to `.env` and configure:
- `DATABASE_URL`: Your database connection
- `API_KEY`: Your API key

## Usage
```python
from myproject import main
main.run()
```

## Project Structure
```
project/
├── module1/        # Description
├── module2/        # Description
└── tests/          # Test suite
```

## Development
```bash
pip install -r requirements-dev.txt
pytest
```

## Contributing
How to contribute

## License
License information
```

**Solutions**:
1. **Auto-generate from code**
   ```bash
   pip install pydoc-markdown
   pydoc-markdown --render-toc > README.md
   ```

2. **Use templates**
   - [Standard Readme](https://github.com/RichardLitt/standard-readme)
   - [Awesome README](https://github.com/matiassingers/awesome-readme)

---

## Type Consistency

### Issue: Inconsistent data types
**Description**: Same data represented differently across modules.

**Examples**:
```python
# Bad - Inconsistent
def func1() -> str:  # Returns "2024-01-01"
    return date.today().isoformat()

def func2(date_obj: datetime):  # Expects datetime object
    pass

# Good - Consistent
def func1() -> datetime:
    return datetime.now()

def func2(date_obj: datetime):
    pass
```

**Solutions**:
1. **Define data models**
   ```python
   from pydantic import BaseModel
   from datetime import datetime
   
   class UserData(BaseModel):
       id: int
       name: str
       created_at: datetime
   ```

2. **Use type aliases**
   ```python
   from typing import NewType
   
   UserId = NewType('UserId', int)
   DateString = NewType('DateString', str)
   ```

3. **Validation decorators**
   ```python
   from functools import wraps
   
   def validate_types(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           hints = get_type_hints(func)
           # Validate types
           return func(*args, **kwargs)
       return wrapper
   ```
