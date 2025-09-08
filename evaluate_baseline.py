"""
Evaluate the baseline model using the provided metrics.
"""

from datasets import load_dataset
from baseline_model import create_baseline_predictions
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best
import random


def evaluate_baseline_model(num_samples: int = 50):
    """
    Evaluate the baseline model on a subset of the test dataset.
    """
    print("Loading test dataset...")
    ds = load_dataset('CADCODER/GenCAD-Code', split='test')
    
    # Use a random subset for evaluation
    random.seed(42)  # For reproducibility
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    
    print(f"Evaluating on {len(indices)} samples...")
    
    # Create baseline predictions
    predictions = {}
    ground_truth = {}
    
    for i, idx in enumerate(indices):
        sample = ds[idx]
        image = sample['image']
        
        # Generate baseline prediction
        from baseline_model import BaselineCadQueryGenerator
        generator = BaselineCadQueryGenerator()
        predicted_code = generator.generate(image)
        
        # Store predictions and ground truth
        sample_id = f"sample_{i:06d}"
        predictions[sample_id] = predicted_code
        ground_truth[sample_id] = sample['cadquery']
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{len(indices)} predictions")
    
    print("\nEvaluating Valid Syntax Rate...")
    vsr = evaluate_syntax_rate_simple(predictions)
    print(f"Valid Syntax Rate: {vsr:.3f}")
    
    print("\nEvaluating IOU (this may take a while)...")
    ious = []
    successful_pairs = 0
    
    for sample_id in predictions.keys():
        try:
            # Calculate IOU between prediction and ground truth
            iou = get_iou_best(predictions[sample_id], ground_truth[sample_id])
            ious.append(iou)
            successful_pairs += 1
            
            if successful_pairs % 10 == 0:
                print(f"Calculated IOU for {successful_pairs} pairs...")
                
        except Exception as e:
            print(f"Error calculating IOU for {sample_id}: {e}")
            continue
    
    if ious:
        mean_iou = sum(ious) / len(ious)
        print(f"Mean IOU: {mean_iou:.3f}")
        print(f"Successful IOU calculations: {len(ious)}/{len(predictions)}")
    else:
        print("No successful IOU calculations")
        mean_iou = 0.0
    
    return {
        'valid_syntax_rate': vsr,
        'mean_iou': mean_iou,
        'successful_pairs': len(ious),
        'total_pairs': len(predictions)
    }


if __name__ == "__main__":
    print("=== BASELINE MODEL EVALUATION ===")
    results = evaluate_baseline_model(num_samples=50)
    
    print("\n=== RESULTS ===")
    print(f"Valid Syntax Rate: {results['valid_syntax_rate']:.3f}")
    print(f"Mean IOU: {results['mean_iou']:.3f}")
    print(f"Successful IOU calculations: {results['successful_pairs']}/{results['total_pairs']}")
    
    # Save results
    with open('baseline_results.txt', 'w') as f:
        f.write("Baseline Model Results\n")
        f.write("=====================\n")
        f.write(f"Valid Syntax Rate: {results['valid_syntax_rate']:.3f}\n")
        f.write(f"Mean IOU: {results['mean_iou']:.3f}\n")
        f.write(f"Successful IOU calculations: {results['successful_pairs']}/{results['total_pairs']}\n")
    
    print("\nResults saved to baseline_results.txt")
