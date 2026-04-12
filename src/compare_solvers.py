#!/usr/bin/env python3
"""
Compare V1 and V2 Solvers
Runs both solvers on same puzzles and compares results
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class SolverComparison:
    def __init__(self, arc_data_dir: str):
        self.arc_data_dir = arc_data_dir
        self.results = {
            'v1': [],
            'v2': []
        }
        
    def get_puzzle_files(self, dataset: str = 'training', count: int = 20) -> List[str]:
        """Get random puzzle files from the dataset"""
        data_path = Path(self.arc_data_dir) / 'data' / dataset
        
        if not data_path.exists():
            # Try alternative path
            data_path = Path(self.arc_data_dir) / 'ARC-AGI-2' / 'data' / dataset
        
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find dataset at {data_path}")
        
        puzzle_files = list(data_path.glob('*.json'))
        
        if len(puzzle_files) < count:
            print(f"Warning: Only found {len(puzzle_files)} puzzles, using all of them")
            return [str(f) for f in puzzle_files]
        
        selected = random.sample(puzzle_files, count)
        return [str(f) for f in selected]
    
    def run_solver(self, solver_script: str, puzzle_file: str) -> Dict:
        """Run a solver on a puzzle file and return results"""
        start_time = time.time()
        
        try:
            # Run the solver
            result = subprocess.run(
                [sys.executable, solver_script, puzzle_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per puzzle
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse the booklet JSON to get accuracy
            puzzle_name = Path(puzzle_file).stem
            version_suffix = '_v2' if 'v2' in solver_script else ''
            booklet_path = Path('test') / f"{puzzle_name}_booklet{version_suffix}.json"
            
            if booklet_path.exists():
                with open(booklet_path, 'r') as f:
                    booklet_data = json.load(f)
                
                return {
                    'puzzle': puzzle_name,
                    'success': True,
                    'accuracy': booklet_data.get('accuracy', 0.0),
                    'time': elapsed_time,
                    'steps': len(booklet_data.get('steps', [])),
                    'prediction_shape': booklet_data.get('final_prediction_shape'),
                    'expected_shape': booklet_data.get('actual_output_shape'),
                    'error': None
                }
            else:
                return {
                    'puzzle': puzzle_name,
                    'success': False,
                    'accuracy': 0.0,
                    'time': elapsed_time,
                    'steps': 0,
                    'prediction_shape': None,
                    'expected_shape': None,
                    'error': 'No booklet generated'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'puzzle': Path(puzzle_file).stem,
                'success': False,
                'accuracy': 0.0,
                'time': 300,
                'steps': 0,
                'prediction_shape': None,
                'expected_shape': None,
                'error': 'Timeout (5 min)'
            }
        except Exception as e:
            return {
                'puzzle': Path(puzzle_file).stem,
                'success': False,
                'accuracy': 0.0,
                'time': time.time() - start_time,
                'steps': 0,
                'prediction_shape': None,
                'expected_shape': None,
                'error': str(e)
            }
    
    def run_comparison(self, puzzle_files: List[str], parallel: bool = True):
        """Run both solvers on all puzzle files"""
        v1_script = 'arc-booklets-solver-v1.py'
        v2_script = 'arc-booklets-solver-v2-stepwise.py'
        
        total = len(puzzle_files)
        
        if parallel:
            print(f"\n🚀 Running in PARALLEL mode (both solvers simultaneously)\n")
            self._run_parallel(puzzle_files, v1_script, v2_script)
        else:
            print(f"\n⏭️ Running in SEQUENTIAL mode (one solver at a time)\n")
            self._run_sequential(puzzle_files, v1_script, v2_script)
    
    def _run_parallel(self, puzzle_files: List[str], v1_script: str, v2_script: str):
        """Run both solvers in parallel using threads"""
        total = len(puzzle_files)
        completed = 0
        lock = threading.Lock()
        
        def run_both_solvers(puzzle_file):
            """Run both V1 and V2 on a single puzzle"""
            puzzle_name = Path(puzzle_file).stem
            
            with lock:
                nonlocal completed
                completed += 1
                print(f"\n{'='*80}")
                print(f"[{completed}/{total}] Starting: {puzzle_name}")
                print(f"{'='*80}")
            
            # Run both in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                v1_future = executor.submit(self.run_solver, v1_script, puzzle_file)
                v2_future = executor.submit(self.run_solver, v2_script, puzzle_file)
                
                v1_result = v1_future.result()
                v2_result = v2_future.result()
            
            with lock:
                self.results['v1'].append(v1_result)
                self.results['v2'].append(v2_result)
                
                print(f"\n📊 Results for {puzzle_name}:")
                print(f"   🔵 V1: {v1_result['accuracy']*100:>5.1f}% | {v1_result['time']:>6.1f}s | {v1_result['steps']:>3} steps")
                print(f"   🟢 V2: {v2_result['accuracy']*100:>5.1f}% | {v2_result['time']:>6.1f}s | {v2_result['steps']:>3} steps")
                
                if v1_result['accuracy'] > v2_result['accuracy']:
                    print(f"   ⚡ V1 wins (+{(v1_result['accuracy']-v2_result['accuracy'])*100:.1f}%)")
                elif v2_result['accuracy'] > v1_result['accuracy']:
                    print(f"   ⚡ V2 wins (+{(v2_result['accuracy']-v1_result['accuracy'])*100:.1f}%)")
                else:
                    print(f"   🤝 Tie ({v1_result['accuracy']*100:.1f}%)")
        
        # Run all puzzles with thread pool
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_both_solvers, pf) for pf in puzzle_files]
            for future in as_completed(futures):
                future.result()  # Wait for completion
    
    def _run_sequential(self, puzzle_files: List[str], v1_script: str, v2_script: str):
        """Run solvers sequentially (one at a time)"""
        total = len(puzzle_files)
        
        for idx, puzzle_file in enumerate(puzzle_files, 1):
            puzzle_name = Path(puzzle_file).stem
            print(f"\n{'='*80}")
            print(f"Puzzle {idx}/{total}: {puzzle_name}")
            print(f"{'='*80}")
            
            # Run V1
            print(f"\n🔵 Running V1 (Holistic)...")
            v1_result = self.run_solver(v1_script, puzzle_file)
            self.results['v1'].append(v1_result)
            print(f"   Accuracy: {v1_result['accuracy']*100:.1f}%  |  Time: {v1_result['time']:.1f}s  |  Steps: {v1_result['steps']}")
            
            # Run V2
            print(f"\n🟢 Running V2 (Step-by-Step)...")
            v2_result = self.run_solver(v2_script, puzzle_file)
            self.results['v2'].append(v2_result)
            print(f"   Accuracy: {v2_result['accuracy']*100:.1f}%  |  Time: {v2_result['time']:.1f}s  |  Steps: {v2_result['steps']}")
            
            # Quick comparison
            if v1_result['accuracy'] > v2_result['accuracy']:
                print(f"   ⚡ V1 wins (+{(v1_result['accuracy']-v2_result['accuracy'])*100:.1f}%)")
            elif v2_result['accuracy'] > v1_result['accuracy']:
                print(f"   ⚡ V2 wins (+{(v2_result['accuracy']-v1_result['accuracy'])*100:.1f}%)")
            else:
                print(f"   🤝 Tie ({v1_result['accuracy']*100:.1f}%)")
    
    def print_summary(self):
        """Print comparison summary"""
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        v1_results = self.results['v1']
        v2_results = self.results['v2']
        
        # Overall stats
        v1_avg_acc = sum(r['accuracy'] for r in v1_results) / len(v1_results) if v1_results else 0
        v2_avg_acc = sum(r['accuracy'] for r in v2_results) / len(v2_results) if v2_results else 0
        
        v1_perfect = sum(1 for r in v1_results if r['accuracy'] == 1.0)
        v2_perfect = sum(1 for r in v2_results if r['accuracy'] == 1.0)
        
        v1_avg_time = sum(r['time'] for r in v1_results) / len(v1_results) if v1_results else 0
        v2_avg_time = sum(r['time'] for r in v2_results) / len(v2_results) if v2_results else 0
        
        print(f"{'Metric':<30} {'V1 (Holistic)':<20} {'V2 (Step-by-Step)':<20}")
        print(f"{'-'*70}")
        print(f"{'Average Accuracy':<30} {v1_avg_acc*100:>18.1f}% {v2_avg_acc*100:>18.1f}%")
        print(f"{'Perfect Predictions':<30} {v1_perfect:>18} / {len(v1_results):<3} {v2_perfect:>18} / {len(v2_results):<3}")
        print(f"{'Average Time (seconds)':<30} {v1_avg_time:>18.1f} {v2_avg_time:>18.1f}")
        
        # Head-to-head
        print(f"\n{'='*80}")
        print("HEAD-TO-HEAD RESULTS")
        print(f"{'='*80}\n")
        
        v1_wins = 0
        v2_wins = 0
        ties = 0
        
        for v1_r, v2_r in zip(v1_results, v2_results):
            if v1_r['accuracy'] > v2_r['accuracy']:
                v1_wins += 1
            elif v2_r['accuracy'] > v1_r['accuracy']:
                v2_wins += 1
            else:
                ties += 1
        
        total = len(v1_results)
        print(f"V1 Wins:   {v1_wins:>3} / {total}  ({v1_wins/total*100:.1f}%)")
        print(f"V2 Wins:   {v2_wins:>3} / {total}  ({v2_wins/total*100:.1f}%)")
        print(f"Ties:      {ties:>3} / {total}  ({ties/total*100:.1f}%)")
        
        # Detailed results
        print(f"\n{'='*80}")
        print("DETAILED RESULTS")
        print(f"{'='*80}\n")
        
        print(f"{'Puzzle':<20} {'V1 Acc %':<10} {'V2 Acc %':<10} {'Winner':<10} {'V1 Time':<10} {'V2 Time':<10}")
        print(f"{'-'*80}")
        
        for v1_r, v2_r in zip(v1_results, v2_results):
            winner = ''
            if v1_r['accuracy'] > v2_r['accuracy']:
                winner = 'V1 ⚡'
            elif v2_r['accuracy'] > v1_r['accuracy']:
                winner = 'V2 ⚡'
            else:
                winner = 'Tie 🤝'
            
            print(f"{v1_r['puzzle']:<20} {v1_r['accuracy']*100:>8.1f}% {v2_r['accuracy']*100:>9.1f}% {winner:<10} {v1_r['time']:>8.1f}s {v2_r['time']:>8.1f}s")
        
        # Save detailed results to JSON
        results_file = f"comparison_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'v1': v1_results,
                'v2': v2_results,
                'summary': {
                    'v1_avg_accuracy': v1_avg_acc,
                    'v2_avg_accuracy': v2_avg_acc,
                    'v1_perfect_count': v1_perfect,
                    'v2_perfect_count': v2_perfect,
                    'v1_wins': v1_wins,
                    'v2_wins': v2_wins,
                    'ties': ties,
                    'v1_avg_time': v1_avg_time,
                    'v2_avg_time': v2_avg_time
                }
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Conclusion
        print(f"\n{'='*80}")
        print("CONCLUSION")
        print(f"{'='*80}\n")
        
        if v2_avg_acc > v1_avg_acc:
            improvement = (v2_avg_acc - v1_avg_acc) * 100
            print(f"✅ V2 (Step-by-Step) performs BETTER by {improvement:.1f}% on average")
        elif v1_avg_acc > v2_avg_acc:
            improvement = (v1_avg_acc - v2_avg_acc) * 100
            print(f"✅ V1 (Holistic) performs BETTER by {improvement:.1f}% on average")
        else:
            print(f"🤝 Both approaches perform EQUALLY well")
        
        print(f"\n{'='*80}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_solvers.py <arc_data_dir> [num_puzzles] [--sequential]")
        print("Example: python compare_solvers.py ARC-AGI-2 5")
        print("Example: python compare_solvers.py ARC-AGI-2 20 --sequential")
        print("\nBy default, runs in PARALLEL mode (faster)")
        print("Use --sequential to run one solver at a time (easier to debug)")
        sys.exit(1)
    
    arc_data_dir = sys.argv[1]
    num_puzzles = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '--sequential' else 20
    parallel = '--sequential' not in sys.argv
    
    print(f"\n{'='*80}")
    print(f"SOLVER COMPARISON: V1 vs V2")
    print(f"{'='*80}\n")
    print(f"ARC Data Directory: {arc_data_dir}")
    print(f"Number of Puzzles: {num_puzzles}")
    print(f"Dataset: training")
    print(f"Execution Mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    print(f"\n{'='*80}\n")
    
    comparison = SolverComparison(arc_data_dir)
    
    # Get puzzle files
    puzzle_files = comparison.get_puzzle_files('training', num_puzzles)
    print(f"Selected {len(puzzle_files)} puzzles\n")
    
    # Run comparison
    comparison.run_comparison(puzzle_files, parallel=parallel)
    
    # Print summary
    comparison.print_summary()


if __name__ == "__main__":
    main()

