#!/usr/bin/env python3
"""
Pipeline Consistency Checker
Validates that pipeline stages have matching input/output data types.
"""
import os
import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class PipelineChecker:
    """Check data flow consistency between pipeline stages."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.pipeline_info = {}
        self.mismatches = []
        
    def analyze_python_function(self, file_path: Path, function_name: str) -> Dict[str, Any]:
        """Analyze a Python function to extract input/output types."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Extract type hints
                    input_types = {}
                    for arg in node.args.args:
                        if arg.annotation:
                            input_types[arg.arg] = ast.unparse(arg.annotation)
                    
                    output_type = None
                    if node.returns:
                        output_type = ast.unparse(node.returns)
                    
                    return {
                        "inputs": input_types,
                        "output": output_type,
                        "docstring": ast.get_docstring(node)
                    }
        except Exception as e:
            return {"error": str(e)}
        
        return {}
    
    def find_pipeline_stages(self) -> List[Dict]:
        """Find pipeline stages in the codebase."""
        stages = []
        
        # Look for common pipeline patterns
        pipeline_patterns = [
            "**/pipeline*.py",
            "**/stage*.py",
            "**/*_stage.py",
            "**/process*.py"
        ]
        
        for pattern in pipeline_patterns:
            for file_path in self.root_dir.glob(pattern):
                stages.append({
                    "file": file_path,
                    "name": file_path.stem
                })
        
        return stages
    
    def check_data_flow(self, stages: List[Dict]) -> List[Dict]:
        """Check if data flows correctly between stages."""
        mismatches = []
        
        # Sort stages by name (assuming sequential naming)
        sorted_stages = sorted(stages, key=lambda x: x['name'])
        
        for i in range(len(sorted_stages) - 1):
            current_stage = sorted_stages[i]
            next_stage = sorted_stages[i + 1]
            
            # Analyze both stages
            current_info = self.analyze_stage_file(current_stage['file'])
            next_info = self.analyze_stage_file(next_stage['file'])
            
            # Check if output of current matches input of next
            if current_info.get('output') and next_info.get('inputs'):
                current_output = current_info['output']
                next_inputs = next_info['inputs']
                
                # Simple type matching (can be enhanced)
                match_found = False
                for input_name, input_type in next_inputs.items():
                    if self._types_compatible(current_output, input_type):
                        match_found = True
                        break
                
                if not match_found:
                    mismatches.append({
                        "stage1": current_stage['name'],
                        "stage2": next_stage['name'],
                        "output_type": current_output,
                        "expected_input": next_inputs,
                        "suggestion": f"Convert {current_output} to match {list(next_inputs.values())[0] if next_inputs else 'unknown'}"
                    })
        
        return mismatches
    
    def analyze_stage_file(self, file_path: Path) -> Dict:
        """Analyze a pipeline stage file."""
        result = {
            "inputs": {},
            "output": None,
            "functions": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "inputs": {},
                        "output": None
                    }
                    
                    # Get input types
                    for arg in node.args.args:
                        if arg.arg != 'self':
                            type_hint = "Any"
                            if arg.annotation:
                                type_hint = ast.unparse(arg.annotation)
                            func_info["inputs"][arg.arg] = type_hint
                    
                    # Get return type
                    if node.returns:
                        func_info["output"] = ast.unparse(node.returns)
                    
                    result["functions"].append(func_info)
                    
                    # Assume main processing function is the longest or has 'process' in name
                    if 'process' in node.name.lower() or 'run' in node.name.lower():
                        result["inputs"] = func_info["inputs"]
                        result["output"] = func_info["output"]
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two types are compatible."""
        if not type1 or not type2:
            return True  # Can't determine, assume compatible
        
        # Normalize types
        type1 = type1.strip().lower()
        type2 = type2.strip().lower()
        
        # Exact match
        if type1 == type2:
            return True
        
        # Generic types
        if 'any' in type1 or 'any' in type2:
            return True
        
        # Check for container types (List, Dict, etc.)
        if type1.startswith('list') and type2.startswith('list'):
            return True
        if type1.startswith('dict') and type2.startswith('dict'):
            return True
        
        return False
    
    def generate_report(self) -> Dict:
        """Generate a report of pipeline consistency issues."""
        stages = self.find_pipeline_stages()
        
        if not stages:
            return {
                "status": "no_pipelines",
                "message": "No pipeline stages found"
            }
        
        mismatches = self.check_data_flow(stages)
        
        return {
            "status": "ok" if not mismatches else "issues_found",
            "total_stages": len(stages),
            "stages": [s['name'] for s in stages],
            "mismatches": mismatches
        }
    
    def print_report(self, report: Dict):
        """Print formatted report."""
        print("\n" + "="*60)
        print("🔗 PIPELINE CONSISTENCY REPORT")
        print("="*60)
        
        if report["status"] == "no_pipelines":
            print("\n✅ No pipeline stages detected")
            return
        
        print(f"\n📊 Found {report['total_stages']} pipeline stages:")
        for stage in report['stages']:
            print(f"  • {stage}")
        
        if report.get('mismatches'):
            print(f"\n⚠️  Found {len(report['mismatches'])} data flow mismatches:")
            for i, mismatch in enumerate(report['mismatches'], 1):
                print(f"\n  {i}. {mismatch['stage1']} → {mismatch['stage2']}")
                print(f"     Output: {mismatch['output_type']}")
                print(f"     Expected Input: {mismatch['expected_input']}")
                print(f"     💡 Suggestion: {mismatch['suggestion']}")
        else:
            print("\n✅ All pipeline stages have compatible data flow!")
        
        print("\n" + "="*60)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python check_pipeline.py <directory>")
        print("Example: python check_pipeline.py /path/to/project")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    
    if not os.path.isdir(root_dir):
        print(f"❌ Error: {root_dir} is not a valid directory")
        sys.exit(1)
    
    checker = PipelineChecker(root_dir)
    report = checker.generate_report()
    checker.print_report(report)
    
    # Save report
    report_path = Path(root_dir) / "pipeline_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
