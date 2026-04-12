#!/usr/bin/env python3
"""
ARC Booklet Ensemble (Option 2: Multi-Booklet Synthesis)
Creates separate booklets for each training example, then synthesizes
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import subprocess
import shutil

class BookletEnsemble:
    def __init__(self):
        self.task_data = None
        self.task_name = None
        
    def load_task(self, file_path: str) -> Dict[str, Any]:
        """Load an ARC-AGI task from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def run_generator_on_example(self, example_input, example_output, example_num, output_dir):
        """Create a temporary task file with single example and run generator"""
        # Create temp task with just this example
        temp_task = {
            "train": [{
                "input": example_input,
                "output": example_output
            }],
            "test": []
        }
        
        temp_file = Path(f"temp_task_ex{example_num}.json")
        with open(temp_file, 'w') as f:
            json.dump(temp_task, f)
        
        try:
            # Run the booklet generator
            result = subprocess.run(
                [sys.executable, "arc-booklet-generator.py", str(temp_file), output_dir],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Clean up temp file
            temp_file.unlink()
            
            if result.returncode == 0:
                # Rename the generated booklet
                booklet_dir = Path(output_dir) / f"temp_task_ex{example_num}_booklet"
                new_name = Path(output_dir) / f"example_{example_num}_booklet"
                
                if booklet_dir.exists():
                    if new_name.exists():
                        shutil.rmtree(new_name)
                    booklet_dir.rename(new_name)
                    
                    # Load metadata
                    metadata_path = new_name / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            return json.load(f), new_name
            
            return None, None
            
        except Exception as e:
            print(f"Error running generator: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return None, None
    
    def find_common_pattern(self, all_steps):
        """Find common patterns across all booklets"""
        if not all_steps:
            return []
        
        # Find steps that appear in all booklets
        common_steps = []
        
        # Get the first set of steps as baseline
        baseline = all_steps[0]
        
        for step_idx, baseline_step in enumerate(baseline):
            # Check if similar step appears in all other booklets
            appears_in_all = True
            
            for other_steps in all_steps[1:]:
                # Simple similarity check (could be more sophisticated)
                found_similar = any(
                    self._steps_similar(baseline_step, other_step)
                    for other_step in other_steps
                )
                
                if not found_similar:
                    appears_in_all = False
                    break
            
            if appears_in_all:
                common_steps.append(baseline_step)
        
        return common_steps
    
    def _steps_similar(self, step1, step2, threshold=0.5):
        """Check if two step descriptions are similar"""
        # Simple word overlap similarity
        words1 = set(step1.lower().split())
        words2 = set(step2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return (overlap / union) > threshold
    
    def create_ensemble(self, task_file: str, output_dir: str = "ensemble_booklets"):
        """Main ensemble process"""
        
        # Load task
        self.task_data = self.load_task(task_file)
        self.task_name = Path(task_file).stem
        
        training_examples = self.task_data['train']
        test_examples = self.task_data.get('test', [])
        
        print(f"\n{'='*80}")
        print(f"BOOKLET ENSEMBLE SYNTHESIS")
        print(f"{'='*80}")
        print(f"Task: {self.task_name}")
        print(f"Training Examples: {len(training_examples)}")
        print(f"Test Examples: {len(test_examples)}")
        print(f"{'='*80}\n")
        
        # Create output directory
        output_path = Path(output_dir) / f"{self.task_name}_ensemble"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate booklets for each example
        booklets_data = []
        all_steps = []
        
        for idx, example in enumerate(training_examples):
            print(f"\n[Generating Booklet {idx + 1}/{len(training_examples)}]")
            print("-" * 80)
            
            metadata, booklet_path = self.run_generator_on_example(
                example['input'],
                example['output'],
                idx + 1,
                output_path
            )
            
            if metadata:
                steps = [step['description'] for step in metadata['steps']]
                all_steps.append(steps)
                
                booklets_data.append({
                    "example_number": idx + 1,
                    "booklet_path": str(booklet_path.relative_to(output_path)),
                    "steps": steps,
                    "total_steps": len(steps),
                    "steps_reaching_target": sum(1 for s in metadata['steps'] if s.get('reached_target', False)),
                    "success": any(s.get('reached_target', False) for s in metadata['steps'])
                })
                
                print(f"[OK] Generated booklet with {len(steps)} steps")
                print(f"  Path: {booklet_path.name}")
                print(f"  Success: {'Yes' if booklets_data[-1]['success'] else 'No'}")
            else:
                print(f"[FAIL] Could not generate booklet for example {idx + 1}")
        
        # Synthesize common pattern
        print(f"\n{'='*80}")
        print("SYNTHESIZING COMMON PATTERN")
        print(f"{'='*80}")
        
        common_steps = self.find_common_pattern(all_steps)
        
        print(f"\nFound {len(common_steps)} common steps across all booklets")
        
        # Create synthesis meta-booklet
        synthesis = {
            "task_name": self.task_name,
            "booklets": booklets_data,
            "common_steps": common_steps,
            "total_training_examples": len(training_examples),
            "total_booklets_generated": len(booklets_data),
            "generated_at": datetime.now().isoformat()
        }
        
        # Save synthesis
        synthesis_path = output_path / "synthesis_meta.json"
        with open(synthesis_path, 'w', encoding='utf-8') as f:
            json.dump(synthesis, f, indent=2)
        
        # Generate README
        readme = self._generate_readme(synthesis)
        readme_path = output_path / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme)
        
        # Generate comparison HTML
        html = self._generate_comparison_html(synthesis)
        html_path = output_path / "comparison.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n[SUCCESS] Ensemble synthesis saved to: {output_path}")
        print(f"  - synthesis_meta.json: Complete synthesis data")
        print(f"  - README.txt: Human-readable summary")
        print(f"  - comparison.html: Visual comparison of all booklets")
        print(f"  - example_N_booklet/: Individual booklets for each training example")
        
        return synthesis
    
    def _generate_readme(self, synthesis):
        """Generate README for ensemble"""
        readme = f"""# ARC-AGI Ensemble Booklet Synthesis
Task: {synthesis['task_name']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This ensemble contains {synthesis['total_booklets_generated']} separate booklets, one for each training example.

## Individual Booklets

"""
        
        for booklet in synthesis['booklets']:
            readme += f"### Example {booklet['example_number']}\n"
            readme += f"- Path: {booklet['booklet_path']}\n"
            readme += f"- Total Steps: {booklet['total_steps']}\n"
            readme += f"- Steps Reaching Target: {booklet['steps_reaching_target']}\n"
            readme += f"- Success: {'Yes' if booklet['success'] else 'No'}\n\n"
            readme += "Steps:\n"
            for i, step in enumerate(booklet['steps'], 1):
                step_preview = step[:80] + '...' if len(step) > 80 else step
                readme += f"  {i}. {step_preview}\n"
            readme += "\n"
        
        readme += f"\n## Common Pattern\n\n"
        readme += f"Steps that appear across all booklets:\n\n"
        
        if synthesis['common_steps']:
            for i, step in enumerate(synthesis['common_steps'], 1):
                readme += f"{i}. {step}\n"
        else:
            readme += "No common steps found across all booklets.\n"
        
        readme += f"\n## Analysis\n\n"
        readme += f"Total booklets: {len(synthesis['booklets'])}\n"
        readme += f"Successful booklets: {sum(1 for b in synthesis['booklets'] if b['success'])}\n"
        readme += f"Common steps found: {len(synthesis['common_steps'])}\n"
        
        return readme
    
    def _generate_comparison_html(self, synthesis):
        """Generate HTML comparison page"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ensemble Comparison - {synthesis['task_name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .booklet {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .booklet h3 {{ margin-top: 0; color: #0066cc; }}
        .steps {{ background: #f5f5f5; padding: 10px; border-radius: 3px; }}
        .success {{ color: green; }}
        .fail {{ color: red; }}
        .common-steps {{ background: #e6f3ff; padding: 15px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Ensemble Booklet Comparison</h1>
    <p><strong>Task:</strong> {synthesis['task_name']}</p>
    <p><strong>Total Booklets:</strong> {synthesis['total_booklets_generated']}</p>
    <p><strong>Generated:</strong> {synthesis['generated_at']}</p>
    
    <div class="common-steps">
        <h2>Common Steps ({len(synthesis['common_steps'])} found)</h2>
        <ol>
"""
        
        for step in synthesis['common_steps']:
            html += f"            <li>{step}</li>\n"
        
        html += """        </ol>
    </div>
    
    <h2>Individual Booklets</h2>
"""
        
        for booklet in synthesis['booklets']:
            success_class = 'success' if booklet['success'] else 'fail'
            success_text = 'Success' if booklet['success'] else 'Failed'
            
            html += f"""    <div class="booklet">
        <h3>Example {booklet['example_number']} - <span class="{success_class}">{success_text}</span></h3>
        <p><strong>Path:</strong> {booklet['booklet_path']}</p>
        <p><strong>Steps:</strong> {booklet['total_steps']} | <strong>Reached Target:</strong> {booklet['steps_reaching_target']}</p>
        <div class="steps">
            <ol>
"""
            
            for step in booklet['steps']:
                html += f"                <li>{step}</li>\n"
            
            html += """            </ol>
        </div>
    </div>
"""
        
        html += """</body>
</html>"""
        
        return html


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc-booklet-ensemble.py <task_json_file> [output_dir]")
        print("Example: python arc-booklet-ensemble.py ../saturn-arc/ARC-AGI-2/ARC-AGI-1/data/training/00d62c1b.json")
        sys.exit(1)
    
    task_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "ensemble_booklets"
    
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        ensemble = BookletEnsemble()
        ensemble.create_ensemble(task_file, output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

