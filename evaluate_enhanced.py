"""
Evaluate the robust enhanced model using the provided metrics.
"""

from datasets import load_dataset
from enhanced_model import create_robust_predictions
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best
import random
import time


def evaluate_robust_model(num_samples: int = 50):
    """
    Evaluate the robust enhanced model on a subset of the test dataset.
    """
    print("Loading test dataset...")
    ds = load_dataset('CADCODER/GenCAD-Code', split='test')
    
    # Use a random subset for evaluation
    random.seed(42)  # For reproducibility
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    
    print(f"Evaluating on {len(indices)} samples...")
    
    # Create robust enhanced predictions
    predictions = {}
    ground_truth = {}
    
    start_time = time.time()
    
    for i, idx in enumerate(indices):
        sample = ds[idx]
        image = sample['image']
        
        # Generate robust enhanced prediction
        from enhanced_model import RobustEnhancedCadQueryGenerator
        generator = RobustEnhancedCadQueryGenerator()
        
        # Learn from a subset of training data first
        if i == 0:  # Only learn once
            print("Learning enhanced patterns from training data...")
            train_ds = load_dataset('CADCODER/GenCAD-Code', split='train[:2000]')
            generator.learn_from_dataset(train_ds, num_samples=2000)
        
        predicted_code = generator.generate(image)
        
        # Store predictions and ground truth
        sample_id = f"sample_{i:06d}"
        predictions[sample_id] = predicted_code
        ground_truth[sample_id] = sample['cadquery']
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Generated {i + 1}/{len(indices)} predictions (elapsed: {elapsed:.1f}s)")
    
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
        max_iou = max(ious)
        min_iou = min(ious)
        std_iou = (sum((x - mean_iou) ** 2 for x in ious) / len(ious)) ** 0.5
        
        print(f"Mean IOU: {mean_iou:.3f}")
        print(f"Max IOU: {max_iou:.3f}")
        print(f"Min IOU: {min_iou:.3f}")
        print(f"Std IOU: {std_iou:.3f}")
        print(f"Successful IOU calculations: {len(ious)}/{len(predictions)}")
    else:
        print("No successful IOU calculations")
        mean_iou = 0.0
        max_iou = 0.0
        min_iou = 0.0
        std_iou = 0.0
    
    return {
        'valid_syntax_rate': vsr,
        'mean_iou': mean_iou,
        'max_iou': max_iou,
        'min_iou': min_iou,
        'std_iou': std_iou,
        'successful_pairs': len(ious),
        'total_pairs': len(predictions)
    }


if __name__ == "__main__":
    print("=== ROBUST ENHANCED MODEL EVALUATION ===")
    results = evaluate_robust_model(num_samples=50)
    
    print("\n=== RESULTS ===")
    print(f"Valid Syntax Rate: {results['valid_syntax_rate']:.3f}")
    print(f"Mean IOU: {results['mean_iou']:.3f}")
    print(f"Max IOU: {results['max_iou']:.3f}")
    print(f"Min IOU: {results['min_iou']:.3f}")
    print(f"Std IOU: {results['std_iou']:.3f}")
    print(f"Successful IOU calculations: {results['successful_pairs']}/{results['total_pairs']}")
    
    # Save results
    with open('robust_results.txt', 'w') as f:
        f.write("Robust Enhanced Model Results\n")
        f.write("=============================\n")
        f.write(f"Valid Syntax Rate: {results['valid_syntax_rate']:.3f}\n")
        f.write(f"Mean IOU: {results['mean_iou']:.3f}\n")
        f.write(f"Max IOU: {results['max_iou']:.3f}\n")
        f.write(f"Min IOU: {results['min_iou']:.3f}\n")
        f.write(f"Std IOU: {results['std_iou']:.3f}\n")
        f.write(f"Successful IOU calculations: {results['successful_pairs']}/{results['total_pairs']}\n")
    
    print("\nResults saved to robust_results.txt")
