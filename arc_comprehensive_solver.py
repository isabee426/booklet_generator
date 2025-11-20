#!/usr/bin/env python3
"""
ARC Comprehensive Solver
Complete analysis and booklet generation system following the comprehensive guide
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Set
from PIL import Image
import numpy as np
from datetime import datetime
from collections import defaultdict
from openai import OpenAI

# Import visualizer
from arc_visualizer import grid_to_image, ARC_COLORS

class PuzzleAnalyzer:
    """Analyzes puzzles to identify type, patterns, and reference objects"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def get_dimensions(self, grid: List[List[int]]) -> Tuple[int, int]:
        """Get grid dimensions (rows, cols)"""
        if not grid:
            return (0, 0)
        return (len(grid), len(grid[0]) if grid[0] else 0)
    
    def get_size(self, grid: List[List[int]]) -> int:
        """Get total number of cells"""
        rows, cols = self.get_dimensions(grid)
        return rows * cols
    
    def identify_puzzle_type(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Identify puzzle type based on size relationships"""
        size_relationships = []
        
        for ex in train_examples:
            input_size = self.get_size(ex['input'])
            output_size = self.get_size(ex['output'])
            input_dims = self.get_dimensions(ex['input'])
            output_dims = self.get_dimensions(ex['output'])
            
            if input_size > output_size:
                rel = "input_gt_output"
            elif input_size == output_size:
                rel = "input_eq_output"
            else:
                rel = "input_lt_output"
            
            size_relationships.append({
                'relationship': rel,
                'input_dims': input_dims,
                'output_dims': output_dims,
                'input_size': input_size,
                'output_size': output_size
            })
        
        # Determine dominant type
        rel_counts = defaultdict(int)
        for sr in size_relationships:
            rel_counts[sr['relationship']] += 1
        
        dominant_type = max(rel_counts.items(), key=lambda x: x[1])[0]
        
        # Determine if ratio varies
        ratios = []
        for sr in size_relationships:
            if sr['input_size'] > 0:
                ratio = sr['output_size'] / sr['input_size']
                ratios.append(ratio)
        
        ratio_varies = len(set(round(r, 2) for r in ratios)) > 1 if ratios else False
        
        puzzle_type_info = {
            'dominant_type': dominant_type,
            'type_name': {
                'input_gt_output': 'Pattern Extraction/Cropping',
                'input_eq_output': 'Same-Size Transformation',
                'input_lt_output': 'Expansion/Tiling'
            }[dominant_type],
            'size_relationships': size_relationships,
            'ratio_varies': ratio_varies,
            'suggested_initial_grid': self._suggest_initial_grid(train_examples, dominant_type)
        }
        
        return puzzle_type_info
    
    def _suggest_initial_grid(self, train_examples: List[Dict], puzzle_type: str) -> Dict[str, Any]:
        """Suggest initial grid size based on puzzle type"""
        if puzzle_type == "input_gt_output":
            # Usually start with input size, but may need to crop
            input_dims = [self.get_dimensions(ex['input']) for ex in train_examples]
            output_dims = [self.get_dimensions(ex['output']) for ex in train_examples]
            return {
                'start_with': 'input_size',
                'may_need_crop': True,
                'typical_input': input_dims[0] if input_dims else None,
                'typical_output': output_dims[0] if output_dims else None
            }
        elif puzzle_type == "input_eq_output":
            # Start with input size (same as output)
            input_dims = [self.get_dimensions(ex['input']) for ex in train_examples]
            return {
                'start_with': 'input_size',
                'may_need_crop': False,
                'typical_size': input_dims[0] if input_dims else None
            }
        else:  # input_lt_output
            # Start with input size, will expand
            input_dims = [self.get_dimensions(ex['input']) for ex in train_examples]
            output_dims = [self.get_dimensions(ex['output']) for ex in train_examples]
            return {
                'start_with': 'input_size',
                'will_expand': True,
                'typical_input': input_dims[0] if input_dims else None,
                'typical_output': output_dims[0] if output_dims else None
            }
    
    def find_reference_objects(self, train_examples: List[Dict]) -> Optional[Dict[str, Any]]:
        """Find reference objects that stay constant across examples"""
        # Compare inputs to find common objects
        input_objects = []
        output_objects = []
        
        for ex in train_examples:
            input_objs = self._detect_objects_simple(ex['input'])
            output_objs = self._detect_objects_simple(ex['output'])
            input_objects.append(input_objs)
            output_objects.append(output_objs)
        
        # Find objects that appear in same location/color across inputs
        reference_candidates = []
        
        if len(input_objects) > 0:
            first_input_objs = input_objects[0]
            for obj in first_input_objs:
                # Check if similar object exists in other inputs
                matches = 1
                for other_objs in input_objects[1:]:
                    if self._find_similar_object(obj, other_objs):
                        matches += 1
                
                if matches == len(input_objects):
                    reference_candidates.append({
                        'object': obj,
                        'type': 'input_constant',
                        'matches': matches
                    })
        
        # Find objects that appear in same location/color in outputs
        if len(output_objects) > 0:
            first_output_objs = output_objects[0]
            for obj in first_output_objs:
                matches = 1
                for other_objs in output_objects[1:]:
                    if self._find_similar_object(obj, other_objs):
                        matches += 1
                
                if matches == len(output_objects):
                    reference_candidates.append({
                        'object': obj,
                        'type': 'output_constant',
                        'matches': matches
                    })
        
        if reference_candidates:
            # Return the most significant reference object
            best_ref = max(reference_candidates, key=lambda x: x['object'].get('size', 0))
            return {
                'object': best_ref['object'],
                'type': best_ref['type'],
                'reasoning': f"Object appears consistently across all {best_ref['matches']} examples"
            }
        
        return None
    
    def _detect_objects_simple(self, grid: List[List[int]]) -> List[Dict]:
        """Simple object detection - finds connected regions"""
        objects = []
        visited = set()
        rows, cols = self.get_dimensions(grid)
        
        def get_connected_region(r, c, color):
            """Get all connected cells of same color"""
            region = []
            stack = [(r, c)]
            
            while stack:
                cr, cc = stack.pop()
                if (cr, cc) in visited or cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    continue
                if grid[cr][cc] != color:
                    continue
                
                visited.add((cr, cc))
                region.append((cr, cc))
                
                # Check neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
            
            return region
        
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid[r][c] != 0:
                    color = grid[r][c]
                    region = get_connected_region(r, c, color)
                    
                    if region:
                        min_r = min(p[0] for p in region)
                        max_r = max(p[0] for p in region)
                        min_c = min(p[1] for p in region)
                        max_c = max(p[1] for p in region)
                        
                        objects.append({
                            'bbox': [min_r, min_c, max_r, max_c],
                            'colors': [color],
                            'size': len(region),
                            'cells': region,
                            'description': f"Connected region of color {color}"
                        })
        
        return objects
    
    def _find_similar_object(self, obj1: Dict, objects: List[Dict], threshold: float = 0.8) -> Optional[Dict]:
        """Find similar object in list based on color, size, and position"""
        for obj2 in objects:
            # Check color match
            if obj1['colors'] != obj2['colors']:
                continue
            
            # Check size similarity
            size1 = obj1['size']
            size2 = obj2['size']
            size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0
            
            if size_ratio < threshold:
                continue
            
            # Check position similarity (bbox overlap)
            bbox1 = obj1['bbox']
            bbox2 = obj2['bbox']
            
            # Simple overlap check
            if abs(bbox1[0] - bbox2[0]) < 3 and abs(bbox1[1] - bbox2[1]) < 3:
                return obj2
        
        return None
    
    def analyze_transitions(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Analyze transitions from input to output"""
        transitions = []
        
        for ex in train_examples:
            input_grid = ex['input']
            output_grid = ex['output']
            
            # Compare grids
            similarities = self._compare_grids(input_grid, output_grid)
            differences = self._find_differences(input_grid, output_grid)
            
            transitions.append({
                'similarities': similarities,
                'differences': differences,
                'input_dims': self.get_dimensions(input_grid),
                'output_dims': self.get_dimensions(output_grid)
            })
        
        # Find common patterns across transitions
        common_similarities = self._find_common_elements([t['similarities'] for t in transitions])
        common_differences = self._find_common_elements([t['differences'] for t in transitions])
        
        return {
            'individual_transitions': transitions,
            'common_similarities': common_similarities,
            'common_differences': common_differences,
            'pattern': self._identify_transition_pattern(transitions)
        }
    
    def _compare_grids(self, grid1: List[List[int]], grid2: List[List[int]]) -> Dict[str, Any]:
        """Compare two grids to find similarities"""
        dims1 = self.get_dimensions(grid1)
        dims2 = self.get_dimensions(grid2)
        
        similarities = {
            'same_dimensions': dims1 == dims2,
            'same_colors_present': set() == set(),  # Will be filled
            'structural_similarity': 0.0
        }
        
        # Check colors
        colors1 = set()
        colors2 = set()
        for row in grid1:
            colors1.update(row)
        for row in grid2:
            colors2.update(row)
        
        similarities['same_colors_present'] = colors1 == colors2
        
        # Calculate structural similarity if same size
        if dims1 == dims2:
            matches = 0
            total = dims1[0] * dims1[1]
            for r in range(dims1[0]):
                for c in range(dims1[1]):
                    if grid1[r][c] == grid2[r][c]:
                        matches += 1
            similarities['structural_similarity'] = matches / total if total > 0 else 0
        
        return similarities
    
    def _find_differences(self, grid1: List[List[int]], grid2: List[List[int]]) -> Dict[str, Any]:
        """Find differences between grids"""
        dims1 = self.get_dimensions(grid1)
        dims2 = self.get_dimensions(grid2)
        
        differences = {
            'size_change': dims1 != dims2,
            'color_changes': [],
            'structural_changes': []
        }
        
        if dims1 == dims2:
            # Same size - find color/structural differences
            color_map = {}
            for r in range(dims1[0]):
                for c in range(dims1[1]):
                    if grid1[r][c] != grid2[r][c]:
                        if grid1[r][c] not in color_map:
                            color_map[grid1[r][c]] = set()
                        color_map[grid1[r][c]].add(grid2[r][c])
            
            for from_color, to_colors in color_map.items():
                if len(to_colors) == 1:
                    differences['color_changes'].append({
                        'from': from_color,
                        'to': list(to_colors)[0]
                    })
        
        return differences
    
    def _find_common_elements(self, element_lists: List[List]) -> List:
        """Find elements common to all lists"""
        if not element_lists:
            return []
        
        common = set(element_lists[0])
        for lst in element_lists[1:]:
            common &= set(lst)
        
        return list(common)
    
    def _identify_transition_pattern(self, transitions: List[Dict]) -> str:
        """Identify the overall transition pattern"""
        # Simple pattern identification
        if all(t['similarities']['same_dimensions'] for t in transitions):
            if all(t['differences']['color_changes'] for t in transitions):
                return "color_mapping"
            return "same_size_transform"
        elif all(t['input_dims'][0] < t['output_dims'][0] or t['input_dims'][1] < t['output_dims'][1] for t in transitions):
            return "expansion"
        else:
            return "cropping"
    
    def compare_all_inputs(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Compare all training inputs against each other"""
        inputs = [ex['input'] for ex in train_examples]
        
        # Find common elements
        common_colors = set()
        common_structures = []
        
        if inputs:
            # Colors present in all inputs
            all_colors = [set() for _ in inputs]
            for i, grid in enumerate(inputs):
                for row in grid:
                    all_colors[i].update(row)
            
            common_colors = set.intersection(*all_colors) if all_colors else set()
            
            # Structural similarities
            dims = [self.get_dimensions(grid) for grid in inputs]
            common_dims = len(set(dims)) == 1
        
        # Find differences
        differences = []
        for i in range(len(inputs)):
            for j in range(i+1, len(inputs)):
                diff = self._find_differences(inputs[i], inputs[j])
                differences.append({
                    'example_pair': (i, j),
                    'differences': diff
                })
        
        return {
            'common_colors': list(common_colors),
            'common_dimensions': common_dims if inputs else False,
            'input_differences': differences,
            'num_inputs': len(inputs)
        }
    
    def compare_all_outputs(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Compare all training outputs against each other"""
        outputs = [ex['output'] for ex in train_examples]
        
        # Find common elements
        common_colors = set()
        common_structures = []
        
        if outputs:
            # Colors present in all outputs
            all_colors = [set() for _ in outputs]
            for i, grid in enumerate(outputs):
                for row in grid:
                    all_colors[i].update(row)
            
            common_colors = set.intersection(*all_colors) if all_colors else set()
            
            # Structural similarities
            dims = [self.get_dimensions(grid) for grid in outputs]
            common_dims = len(set(dims)) == 1
        
        # Find exact matches (things that are identical across all outputs)
        exact_matches = []
        if outputs and len(outputs) > 1:
            dims = self.get_dimensions(outputs[0])
            for r in range(dims[0]):
                for c in range(dims[1]):
                    if all(len(out) > r and len(out[r]) > c for out in outputs):
                        colors = [out[r][c] for out in outputs]
                        if len(set(colors)) == 1:
                            exact_matches.append({
                                'position': (r, c),
                                'color': colors[0]
                            })
        
        return {
            'common_colors': list(common_colors),
            'common_dimensions': common_dims if outputs else False,
            'exact_matches': exact_matches,
            'num_outputs': len(outputs)
        }
    
    def compare_input_to_output_individual(self, train_examples: List[Dict]) -> List[Dict]:
        """Compare input to output for each training example individually"""
        comparisons = []
        
        for i, ex in enumerate(train_examples):
            input_grid = ex['input']
            output_grid = ex['output']
            
            # Detailed comparison
            similarities = self._compare_grids(input_grid, output_grid)
            differences = self._find_differences(input_grid, output_grid)
            
            # Object-level comparison
            input_objects = self._detect_objects_simple(input_grid)
            output_objects = self._detect_objects_simple(output_grid)
            
            # Match objects
            object_matches = []
            for in_obj in input_objects:
                best_match = None
                best_score = 0
                for out_obj in output_objects:
                    score = self._object_similarity_score(in_obj, out_obj)
                    if score > best_score:
                        best_score = score
                        best_match = out_obj
                
                object_matches.append({
                    'input_object': in_obj,
                    'output_object': best_match,
                    'similarity_score': best_score
                })
            
            comparisons.append({
                'example_index': i,
                'similarities': similarities,
                'differences': differences,
                'input_objects': input_objects,
                'output_objects': output_objects,
                'object_matches': object_matches,
                'input_dims': self.get_dimensions(input_grid),
                'output_dims': self.get_dimensions(output_grid)
            })
        
        return comparisons
    
    def _object_similarity_score(self, obj1: Dict, obj2: Dict) -> float:
        """Calculate similarity score between two objects"""
        score = 0.0
        
        # Color match
        if obj1['colors'] == obj2['colors']:
            score += 0.4
        
        # Size similarity
        size1 = obj1['size']
        size2 = obj2['size']
        if max(size1, size2) > 0:
            size_ratio = min(size1, size2) / max(size1, size2)
            score += 0.3 * size_ratio
        
        # Position similarity (bbox center distance)
        bbox1 = obj1['bbox']
        bbox2 = obj2['bbox']
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        # Normalize distance (assume max distance of 30)
        pos_score = max(0, 1 - distance / 30)
        score += 0.3 * pos_score
        
        return score
    
    def find_incremental_steps(self, train_examples: List[Dict]) -> List[Dict]:
        """Find incremental steps that build on each other"""
        steps = []
        
        # Analyze how examples build upon each other
        for i in range(len(train_examples) - 1):
            ex1 = train_examples[i]
            ex2 = train_examples[i + 1]
            
            # Compare inputs
            input_diff = self._find_differences(ex1['input'], ex2['input'])
            output_diff = self._find_differences(ex1['output'], ex2['output'])
            
            # Find what new information example 2 provides
            steps.append({
                'from_example': i,
                'to_example': i + 1,
                'input_differences': input_diff,
                'output_differences': output_diff,
                'new_information': self._extract_new_information(input_diff, output_diff)
            })
        
        return steps
    
    def _extract_new_information(self, input_diff: Dict, output_diff: Dict) -> str:
        """Extract what new information is provided"""
        info_parts = []
        
        if input_diff.get('size_change'):
            info_parts.append("Different grid size")
        
        if input_diff.get('color_changes'):
            info_parts.append(f"Different colors: {input_diff['color_changes']}")
        
        if output_diff.get('color_changes'):
            info_parts.append(f"Output color changes: {output_diff['color_changes']}")
        
        return "; ".join(info_parts) if info_parts else "No significant new information"
    
    def analyze_whole_grid_patterns(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns using the whole grid"""
        patterns = {
            'color_patterns': [],
            'shape_patterns': [],
            'negative_space_patterns': [],
            'common_cells': [],
            'divided_sections': []
        }
        
        for ex in train_examples:
            input_grid = ex['input']
            output_grid = ex['output']
            
            # Color patterns
            input_colors = defaultdict(int)
            output_colors = defaultdict(int)
            for row in input_grid:
                for cell in row:
                    input_colors[cell] += 1
            for row in output_grid:
                for cell in row:
                    output_colors[cell] += 1
            
            color_transitions = {}
            for color in input_colors:
                if color in output_colors:
                    color_transitions[color] = output_colors[color] - input_colors[color]
            
            patterns['color_patterns'].append({
                'input_colors': dict(input_colors),
                'output_colors': dict(output_colors),
                'transitions': color_transitions
            })
            
            # Negative space (color 0)
            input_negative = input_colors.get(0, 0)
            output_negative = output_colors.get(0, 0)
            patterns['negative_space_patterns'].append({
                'input_negative': input_negative,
                'output_negative': output_negative,
                'change': output_negative - input_negative
            })
            
            # Common cells (same position, same color)
            if self.get_dimensions(input_grid) == self.get_dimensions(output_grid):
                common = 0
                total = 0
                for r in range(len(input_grid)):
                    for c in range(len(input_grid[r])):
                        total += 1
                        if input_grid[r][c] == output_grid[r][c]:
                            common += 1
                
                patterns['common_cells'].append({
                    'common_count': common,
                    'total_count': total,
                    'ratio': common / total if total > 0 else 0
                })
        
        return patterns
    
    def generate_rule(self, puzzle_type: Dict, reference_objects: Optional[Dict], 
                     transitions: Dict, input_comparison: Dict, output_comparison: Dict,
                     individual_comparisons: List[Dict]) -> str:
        """Generate 3-5 sentence general rule with comprehensive analysis"""
        rule_parts = []
        
        # Puzzle type
        rule_parts.append(f"This puzzle involves {puzzle_type['type_name'].lower()}.")
        
        # Size relationship
        if puzzle_type['dominant_type'] == 'input_eq_output':
            rule_parts.append("The input and output grids have the same dimensions.")
        elif puzzle_type['dominant_type'] == 'input_gt_output':
            rule_parts.append("The output is smaller than the input, requiring extraction or cropping.")
        else:
            rule_parts.append("The output is larger than the input, requiring expansion or tiling.")
        
        # Reference objects
        if reference_objects:
            rule_parts.append(f"A reference object (color {reference_objects['object']['colors'][0]}) remains constant and guides the transformation.")
        
        # Common output elements
        if output_comparison.get('exact_matches'):
            num_matches = len(output_comparison['exact_matches'])
            rule_parts.append(f"All outputs share {num_matches} identical cell positions, indicating fixed reference points.")
        
        # Transition pattern
        pattern = transitions.get('pattern', 'unknown')
        if pattern == 'color_mapping':
            color_changes = []
            for comp in individual_comparisons:
                for change in comp['differences'].get('color_changes', []):
                    color_changes.append(f"color {change['from']} to color {change['to']}")
            if color_changes:
                unique_changes = list(set(color_changes))[:3]
                rule_parts.append(f"The transformation involves mapping {', '.join(unique_changes)}.")
        
        # Complete the rule
        while len(rule_parts) < 3:
            rule_parts.append("Objects are transformed based on their properties and relationships.")
        
        return " ".join(rule_parts[:5])  # Max 5 sentences


class StepGenerator:
    """Generates general steps in the required format"""
    
    def __init__(self, analyzer: PuzzleAnalyzer):
        self.analyzer = analyzer
    
    def generate_steps(self, puzzle_type: Dict, reference_objects: Optional[Dict],
                      transitions: Dict, train_examples: List[Dict]) -> List[Dict]:
        """Generate general steps in format: Step x: for each (CONDITION) A, perform transition B"""
        steps = []
        step_num = 1
        
        # Step 1: Initial grid setup
        if puzzle_type['dominant_type'] == 'input_gt_output':
            steps.append({
                'step_number': step_num,
                'instruction': f"Step {step_num}: Start with input grid dimensions {puzzle_type['suggested_initial_grid']['typical_input']}.",
                'condition': 'initial_setup',
                'transition': 'use_input_grid'
            })
            step_num += 1
            
            steps.append({
                'step_number': step_num,
                'instruction': f"Step {step_num}: For each region that matches the output pattern, crop to that region.",
                'condition': 'pattern_match',
                'transition': 'crop_to_pattern'
            })
            step_num += 1
        elif puzzle_type['dominant_type'] == 'input_lt_output':
            steps.append({
                'step_number': step_num,
                'instruction': f"Step {step_num}: Start with input grid dimensions {puzzle_type['suggested_initial_grid']['typical_input']}.",
                'condition': 'initial_setup',
                'transition': 'use_input_grid'
            })
            step_num += 1
            
            steps.append({
                'step_number': step_num,
                'instruction': f"Step {step_num}: For each input pattern, tile it to fill output dimensions {puzzle_type['suggested_initial_grid']['typical_output']}.",
                'condition': 'input_pattern',
                'transition': 'tile_pattern'
            })
            step_num += 1
        else:  # input_eq_output
            steps.append({
                'step_number': step_num,
                'instruction': f"Step {step_num}: Start with input grid dimensions {puzzle_type['suggested_initial_grid']['typical_size']}.",
                'condition': 'initial_setup',
                'transition': 'use_input_grid'
            })
            step_num += 1
        
        # Step 2: Color transformations
        pattern = transitions.get('pattern', '')
        if pattern == 'color_mapping':
            color_changes = []
            for t in transitions['individual_transitions']:
                for change in t['differences'].get('color_changes', []):
                    color_changes.append((change['from'], change['to']))
            
            if color_changes:
                # Group by from color
                from_colors = defaultdict(list)
                for from_c, to_c in color_changes:
                    from_colors[from_c].append(to_c)
                
                for from_color, to_colors_list in from_colors.items():
                    to_color = to_colors_list[0]  # Most common
                    steps.append({
                        'step_number': step_num,
                        'instruction': f"Step {step_num}: For each cell with color {from_color}, change it to color {to_color}.",
                        'condition': f'color_{from_color}',
                        'transition': f'change_to_color_{to_color}'
                    })
                    step_num += 1
        
        # Step 3: Reference object handling
        if reference_objects:
            ref_color = reference_objects['object']['colors'][0]
            steps.append({
                'step_number': step_num,
                'instruction': f"Step {step_num}: Keep the reference object (color {ref_color}) unchanged as it guides the transformation.",
                'condition': f'reference_object_color_{ref_color}',
                'transition': 'preserve_reference'
            })
            step_num += 1
        
        return steps


class BookletGenerator:
    """Generates step-by-step booklets for training examples"""
    
    def __init__(self, api_key: Optional[str] = None, analyzer: Optional[PuzzleAnalyzer] = None):
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.analyzer = analyzer or PuzzleAnalyzer()
        self.tools = self._define_tools()
    
    def _define_tools(self) -> List[Dict]:
        """Define MCP tools for transformations"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_grid",
                    "description": "Generate the resulting grid after applying the action",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "grid": {
                                "type": "array",
                                "description": "2D array representing the grid state after action",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "integer"}
                                }
                            },
                            "visual_analysis": {
                                "type": "string",
                                "description": "1-2 sentences explaining what this step does"
                            }
                        },
                        "required": ["grid", "visual_analysis"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_objects",
                    "description": "Detect all distinct objects in a grid. Each separate filled region is a DISTINCT object, even if same color.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "objects": {
                                "type": "array",
                                "description": "List of detected objects",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "bbox": {
                                            "type": "array",
                                            "description": "Bounding box [min_row, min_col, max_row, max_col]",
                                            "items": {"type": "integer"}
                                        },
                                        "colors": {
                                            "type": "array",
                                            "description": "Primary color(s) of the object",
                                            "items": {"type": "integer"}
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Description of what the object is"
                                        },
                                        "size": {
                                            "type": "integer",
                                            "description": "Number of cells in the object"
                                        }
                                    },
                                    "required": ["bbox", "colors", "description", "size"]
                                }
                            }
                        },
                        "required": ["objects"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "match_objects",
                    "description": "Match objects between input and output grids",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "matches": {
                                "type": "array",
                                "description": "List of matches between input and output objects",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "input_idx": {"type": "integer"},
                                        "output_idx": {"type": "integer", "nullable": True},
                                        "reason": {"type": "string"}
                                    },
                                    "required": ["input_idx", "output_idx", "reason"]
                                }
                            }
                        },
                        "required": ["matches"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "crop",
                    "description": "Crop grid to a specific region",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "grid": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "bbox": {
                                "type": "array",
                                "description": "Bounding box [min_row, min_col, max_row, max_col]",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": ["grid", "bbox"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "transform",
                    "description": "Transform a cropped grid (color changes, rotations, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "grid": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "transformation_type": {
                                "type": "string",
                                "description": "Type of transformation: color_mapping, rotate, flip, etc."
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Transformation-specific parameters"
                            }
                        },
                        "required": ["grid", "transformation_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "uncrop",
                    "description": "Place transformed grid back into full-size grid",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cropped_grid": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "original_grid": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "bbox": {
                                "type": "array",
                                "description": "Original bounding box where cropped grid came from",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": ["cropped_grid", "original_grid", "bbox"]
                    }
                }
            }
        ]
    
    def generate_booklet(self, train_example: Dict, general_steps: List[Dict],
                        puzzle_type: Dict, reference_objects: Optional[Dict]) -> Dict[str, Any]:
        """Generate booklet for a single training example"""
        booklet = {
            'steps': [],
            'input': train_example['input'],
            'output': train_example['output'],
            'current_grid': [row[:] for row in train_example['input']]  # Copy
        }
        
        step_num = 1
        substep_num = 1
        
        for general_step in general_steps:
            # Create substeps for this general step
            substeps = self._create_substeps(
                general_step, 
                booklet['current_grid'],
                train_example,
                puzzle_type,
                reference_objects
            )
            
            for substep in substeps:
                booklet['steps'].append({
                    'step_number': f"{general_step['step_number']}.{substep_num}",
                    'general_step': general_step['step_number'],
                    'instruction': general_step['instruction'],
                    'substep_reasoning': substep.get('reasoning', ''),
                    'grid_before': [row[:] for row in booklet['current_grid']],
                    'grid_after': substep['grid'],
                    'tool_used': substep.get('tool', 'generate_grid'),
                    'tool_params': substep.get('params', {})
                })
                
                booklet['current_grid'] = substep['grid']
                substep_num += 1
            
            substep_num = 1  # Reset for next general step
        
        return booklet
    
    def _create_substeps(self, general_step: Dict, current_grid: List[List[int]],
                        train_example: Dict, puzzle_type: Dict,
                        reference_objects: Optional[Dict]) -> List[Dict]:
        """Create substeps for a general step"""
        substeps = []
        
        condition = general_step.get('condition', '')
        transition = general_step.get('transition', '')
        
        if 'crop' in transition.lower():
            # Find region to crop
            output_dims = self.analyzer.get_dimensions(train_example['output'])
            input_dims = self.analyzer.get_dimensions(current_grid)
            
            if input_dims[0] > output_dims[0] or input_dims[1] > output_dims[1]:
                # Need to crop
                # Simple: crop center region matching output size
                crop_h = min(input_dims[0], output_dims[0])
                crop_w = min(input_dims[1], output_dims[1])
                start_r = (input_dims[0] - crop_h) // 2
                start_c = (input_dims[1] - crop_w) // 2
                
                cropped = [row[start_c:start_c+crop_w] for row in current_grid[start_r:start_r+crop_h]]
                
                substeps.append({
                    'grid': cropped,
                    'tool': 'crop',
                    'params': {'bbox': [start_r, start_c, start_r+crop_h-1, start_c+crop_w-1]},
                    'reasoning': f"Cropped to match output dimensions {output_dims}"
                })
                current_grid = cropped
        
        if 'color' in transition.lower():
            # Apply color transformation
            # Extract color mapping from transition name
            if 'change_to_color' in transition:
                parts = transition.split('_')
                to_color_idx = parts.index('color') + 1
                to_color = int(parts[to_color_idx])
                
                # Find from color from condition
                if 'color_' in condition:
                    from_color = int(condition.split('_')[1])
                    
                    transformed = [row[:] for row in current_grid]
                    for r in range(len(transformed)):
                        for c in range(len(transformed[r])):
                            if transformed[r][c] == from_color:
                                transformed[r][c] = to_color
                    
                    substeps.append({
                        'grid': transformed,
                        'tool': 'transform',
                        'params': {
                            'transformation_type': 'color_mapping',
                            'parameters': {'from_color': from_color, 'to_color': to_color}
                        },
                        'reasoning': f"Changed all cells with color {from_color} to color {to_color}"
                    })
                    current_grid = transformed
        
        if 'tile' in transition.lower():
            # Tile pattern
            input_dims = self.analyzer.get_dimensions(current_grid)
            output_dims = self.analyzer.get_dimensions(train_example['output'])
            
            # Simple tiling: repeat pattern
            tiled = []
            for r in range(output_dims[0]):
                row = []
                for c in range(output_dims[1]):
                    row.append(current_grid[r % input_dims[0]][c % input_dims[1]])
                tiled.append(row)
            
            substeps.append({
                'grid': tiled,
                'tool': 'transform',
                'params': {
                    'transformation_type': 'tiling',
                    'parameters': {'repeat_pattern': True}
                },
                'reasoning': f"Tiled input pattern to fill output dimensions {output_dims}"
            })
            current_grid = tiled
        
        # If no specific transformation, just use current grid
        if not substeps:
            substeps.append({
                'grid': current_grid,
                'tool': 'generate_grid',
                'params': {},
                'reasoning': 'No transformation needed for this step'
            })
        
        return substeps
    
    def analyze(self, train_example: Dict) -> Dict:
        """Analyze a training example (placeholder - would use AI)"""
        # This would call AI to analyze
        return {}


class ARCComprehensiveSolver:
    """Main solver class that orchestrates the entire process"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.analyzer = PuzzleAnalyzer()
        self.step_generator = StepGenerator(self.analyzer)
        self.booklet_generator = BookletGenerator(api_key, self.analyzer)
    
    def solve(self, puzzle_file: str, output_dir: str = "traces") -> Dict[str, Any]:
        """Main solve function"""
        # Load puzzle
        with open(puzzle_file, 'r') as f:
            puzzle_data = json.load(f)
        
        train_examples = puzzle_data.get('train', [])
        test_examples = puzzle_data.get('test', [])
        
        # Step 1: Identify puzzle type
        print("🔍 Identifying puzzle type...")
        puzzle_type = self.analyzer.identify_puzzle_type(train_examples)
        print(f"   Type: {puzzle_type['type_name']}")
        
        # Step 2: Find reference objects
        print("🔍 Finding reference objects...")
        reference_objects = self.analyzer.find_reference_objects(train_examples)
        if reference_objects:
            print(f"   Found reference object: color {reference_objects['object']['colors'][0]}")
        else:
            print("   No reference objects found")
        
        # Step 3: Compare all inputs
        print("🔍 Comparing all inputs...")
        input_comparison = self.analyzer.compare_all_inputs(train_examples)
        print(f"   Common colors: {input_comparison.get('common_colors', [])}")
        
        # Step 4: Compare all outputs
        print("🔍 Comparing all outputs...")
        output_comparison = self.analyzer.compare_all_outputs(train_examples)
        print(f"   Exact matches: {len(output_comparison.get('exact_matches', []))} cells")
        
        # Step 5: Compare input to output for each example
        print("🔍 Comparing input to output for each example...")
        individual_comparisons = self.analyzer.compare_input_to_output_individual(train_examples)
        print(f"   Analyzed {len(individual_comparisons)} examples")
        
        # Step 6: Find incremental steps
        print("🔍 Finding incremental steps...")
        incremental_steps = self.analyzer.find_incremental_steps(train_examples)
        print(f"   Found {len(incremental_steps)} incremental relationships")
        
        # Step 7: Analyze whole grid patterns
        print("🔍 Analyzing whole grid patterns...")
        whole_grid_patterns = self.analyzer.analyze_whole_grid_patterns(train_examples)
        print(f"   Analyzed color, shape, and negative space patterns")
        
        # Step 8: Analyze transitions
        print("🔍 Analyzing transitions...")
        transitions = self.analyzer.analyze_transitions(train_examples)
        print(f"   Pattern: {transitions.get('pattern', 'unknown')}")
        
        # Step 9: Generate rule
        print("📝 Generating rule...")
        rule = self.analyzer.generate_rule(
            puzzle_type, reference_objects, transitions,
            input_comparison, output_comparison, individual_comparisons
        )
        print(f"   Rule: {rule}")
        
        # Step 10: Generate general steps
        print("📝 Generating general steps...")
        general_steps = self.step_generator.generate_steps(
            puzzle_type, reference_objects, transitions, train_examples
        )
        print(f"   Generated {len(general_steps)} general steps")
        
        # Step 11: Generate booklets for training examples
        print("📚 Generating training booklets...")
        training_booklets = []
        for i, ex in enumerate(train_examples):
            print(f"   Processing training example {i+1}/{len(train_examples)}...")
            booklet = self.booklet_generator.generate_booklet(
                ex, general_steps, puzzle_type, reference_objects
            )
            training_booklets.append(booklet)
        
        # Step 12: Generate test prediction
        print("🎯 Generating test prediction...")
        test_booklets = []
        for i, ex in enumerate(test_examples):
            print(f"   Processing test example {i+1}/{len(test_examples)}...")
            # Adapt steps to test input
            test_booklet = self.booklet_generator.generate_booklet(
                ex, general_steps, puzzle_type, reference_objects
            )
            test_booklets.append(test_booklet)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        puzzle_id = os.path.splitext(os.path.basename(puzzle_file))[0]
        
        result = {
            'puzzle_id': puzzle_id,
            'puzzle_type': puzzle_type,
            'reference_objects': reference_objects,
            'input_comparison': input_comparison,
            'output_comparison': output_comparison,
            'individual_comparisons': individual_comparisons,
            'incremental_steps': incremental_steps,
            'whole_grid_patterns': whole_grid_patterns,
            'transitions': transitions,
            'rule': rule,
            'general_steps': general_steps,
            'training_booklets': training_booklets,
            'test_booklets': test_booklets
        }
        
        output_file = os.path.join(output_dir, f"{puzzle_id}_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Saved results to {output_file}")
        
        return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python arc_comprehensive_solver.py <puzzle_file.json> [output_dir]")
        sys.exit(1)
    
    puzzle_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "traces"
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    solver = ARCComprehensiveSolver(api_key=api_key)
    result = solver.solve(puzzle_file, output_dir)

