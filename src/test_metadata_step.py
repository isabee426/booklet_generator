"""Test that step_description is added to all visualization metadata"""
from arc_visual_solver import ARCVisualSolver
from PIL import Image
import json

# Create a test solver instance
solver = ARCVisualSolver()
solver.visualizations_dir = "test_visualizations"
solver.current_task_name = "metadata_test"
solver.current_training_example = 0

# Create a simple test grid
test_grid = [[1, 2], [3, 4]]

# Test 1: Create a hypothetical visualization
from arc_visualizer import grid_to_image
img = grid_to_image(test_grid, 30)

# Add metadata like the real code does
if not img.info:
    img.info = {}
img.info.update({
    'description': "Test hypothetical transformation",
    'visualization_type': "hypothetical",
    'grid_dimensions': "2x2",
    'grid_data': json.dumps(test_grid)
})

# Save it
path = solver.save_visualization(img, "Test hypothetical transformation", visualization_type="hypothetical")
print(f"✅ Saved hypothetical to: {path}")

# Test 2: Create a transform visualization
img2 = grid_to_image(test_grid, 30)
if not img2.info:
    img2.info = {}
img2.info.update({
    'description': "Test transform",
    'visualization_type': "transform",
    'grid_dimensions': "2x2",
    'grid_data': json.dumps(test_grid)
})

path2 = solver.save_visualization(img2, "Test transform", visualization_type="transform")
print(f"✅ Saved transform to: {path2}")

# Now read back the metadata
print("\n" + "="*80)
print("Reading back metadata from hypothetical:")
print("="*80)
metadata1 = solver.get_visualization_metadata(path)
for key, value in metadata1.items():
    print(f"{key}: {value}")

print("\n" + "="*80)
print("Reading back metadata from transform:")
print("="*80)
metadata2 = solver.get_visualization_metadata(path2)
for key, value in metadata2.items():
    print(f"{key}: {value}")

# Check for step_description
print("\n" + "="*80)
print("VERIFICATION:")
print("="*80)
if 'step_description' in metadata1:
    print(f"✅ Hypothetical has step_description: {metadata1['step_description']}")
else:
    print("❌ Hypothetical missing step_description!")

if 'step_description' in metadata2:
    print(f"✅ Transform has step_description: {metadata2['step_description']}")
else:
    print("❌ Transform missing step_description!")

if 'step_number' in metadata1:
    print(f"✅ Hypothetical has step_number: {metadata1['step_number']}")
else:
    print("❌ Hypothetical missing step_number!")

if 'step_number' in metadata2:
    print(f"✅ Transform has step_number: {metadata2['step_number']}")
else:
    print("❌ Transform missing step_number!")
