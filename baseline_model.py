"""
Baseline Model for CadQuery Code Generation
"""

import random
import re
from typing import Dict, List
from PIL import Image
import numpy as np


class BaselineCadQueryGenerator:
    
    def __init__(self):
        # Common templates based on analysis
        self.templates = {
            'simple_box': """
import cadquery as cq

# Simple box
result = cq.Workplane("XY").box(50, 50, 10)
""",
            'box_with_hole': """
import cadquery as cq

# Box with hole
result = (
    cq.Workplane("XY")
    .box(60, 80, 10)
    .faces(">Z")
    .workplane()
    .hole(20)
)
""",
            'cylinder': """
import cadquery as cq

# Simple cylinder
result = cq.Workplane("XY").circle(25).extrude(30)
""",
            'complex_shape': """
import cadquery as cq

# Complex shape with multiple features
result = (
    cq.Workplane("XY")
    .box(100, 80, 15)
    .faces(">Z")
    .workplane()
    .rect(80, 60, forConstruction=True)
    .vertices()
    .hole(8)
    .faces(">Z")
    .workplane()
    .hole(25)
)
""",
            'sketch_based': """
import cadquery as cq

# Sketch-based approach
result = (
    cq.Workplane("XY")
    .moveTo(0, 0)
    .lineTo(50, 0)
    .lineTo(50, 30)
    .lineTo(0, 30)
    .close()
    .extrude(10)
)
"""
        }
        
        self.dimensions = {
            'small': (20, 50),
            'medium': (50, 100), 
            'large': (100, 200)
        }
        
    def analyze_image(self, image: Image.Image) -> Dict:
        """
        Simple image analysis to determine basic properties.
        """
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Basic analysis
        height, width = img_array.shape[:2]
        
        # Simple color analysis
        mean_color = np.mean(img_array, axis=(0, 1))
        
        # Determine if image is mostly dark or light
        brightness = np.mean(mean_color)
        
        # Simple shape detection based on edges (very basic)
        # This is just a placeholder - in reality we'd need more sophisticated analysis
        
        analysis = {
            'size': 'medium' if width > 200 else 'small',
            'brightness': 'light' if brightness > 128 else 'dark',
            'aspect_ratio': width / height,
            'complexity': 'simple'  # Placeholder
        }
        
        return analysis
    
    def select_template(self, analysis: Dict) -> str:
        """
        Select a template based on image analysis.
        """
        # Simple heuristics for template selection
        if analysis['aspect_ratio'] > 1.5:
            return 'simple_box'
        elif analysis['brightness'] == 'light':
            return 'box_with_hole'
        elif analysis['size'] == 'small':
            return 'cylinder'
        else:
            # Random selection for variety
            return random.choice(list(self.templates.keys()))
    
    def customize_template(self, template: str, analysis: Dict) -> str:
        """
        Customize the selected template with random parameters.
        """
        # Get random dimensions
        size_range = self.dimensions[analysis['size']]
        dim1 = random.randint(*size_range)
        dim2 = random.randint(*size_range)
        dim3 = random.randint(5, 20)
        
        # Replace dimensions in template
        customized = template
        
        # Replace common dimension patterns
        customized = re.sub(r'box\(\d+,\s*\d+,\s*\d+\)', 
                          f'box({dim1}, {dim2}, {dim3})', customized)
        customized = re.sub(r'circle\(\d+\)', 
                          f'circle({random.randint(10, 30)})', customized)
        customized = re.sub(r'hole\(\d+\)', 
                          f'hole({random.randint(8, 25)})', customized)
        customized = re.sub(r'extrude\(\d+\)', 
                          f'extrude({random.randint(10, 40)})', customized)
        
        return customized
    
    def generate(self, image: Image.Image) -> str:
        """
        Generate CadQuery code for the given image.
        """
        # Analyze the image
        analysis = self.analyze_image(image)
        
        # Select appropriate template
        template = self.select_template(analysis)
        
        # Customize the template
        code = self.customize_template(self.templates[template], analysis)
        
        return code.strip()


def create_baseline_predictions(dataset, num_samples: int = 100) -> Dict[str, str]:
    """
    Create baseline predictions for a subset of the dataset.
    """
    generator = BaselineCadQueryGenerator()
    predictions = {}
    
    print(f"Generating baseline predictions for {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image']
        
        # Generate prediction
        predicted_code = generator.generate(image)
        
        # Store with sample ID
        sample_id = f"sample_{i:06d}"
        predictions[sample_id] = predicted_code
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} predictions")
    
    return predictions


if __name__ == "__main__":
    # Test the baseline model
    from datasets import load_dataset
    
    print("Loading test dataset...")
    ds = load_dataset('CADCODER/GenCAD-Code', split='test[:10]')
    
    print("Creating baseline predictions...")
    predictions = create_baseline_predictions(ds, num_samples=10)
    
    print(f"\nGenerated {len(predictions)} predictions")
    print("\nFirst prediction:")
    first_key = list(predictions.keys())[0]
    print(f"Sample: {first_key}")
    print(f"Code:\n{predictions[first_key]}")
