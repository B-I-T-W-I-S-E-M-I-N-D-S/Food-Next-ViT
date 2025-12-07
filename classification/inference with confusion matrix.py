import argparse
import torch
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import nextvit

def get_food_classes():
    """Return the list of food categories in your dataset"""
    return [
        'Biriyani', 'Cake', 'Cha', 'Chicken_curry', 'Chicken_wings',
        'Chocolate_cake', 'Chow_mein', 'Crab_Dish_Kakra', 'Doi', 'Fish_Bhuna_Mach_Bhuna',
        'French_fries', 'Fried_fish_Mach_Bhaja', 'Fried_rice', 'Khichuri', 'Meat_Curry_Gosht_Bhuna','Misti',
        'Momos','Salad', 'Sandwich', 'Shik_kabab',
        'Singgara', 'bakorkhani', 'cheesecake', 'cup_cakes', 'fuchka',
        'haleem', 'ice_cream', 'jilapi', 'nehari', 'pakora',
        'pizza', 'poached_egg', 'porota'
    ]

def get_transform(input_size=224):
    """Create the same transform used during validation"""
    size = int((256 / 224) * input_size)
    t = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    return t

def load_model(checkpoint_path, model_name='nextvit_small', num_classes=49, device='cuda'):
    """Load the trained model from checkpoint"""
    # Create model
    model = create_model(
        model_name,
        num_classes=num_classes,
    )
    
    # Load checkpoint with safe globals for argparse.Namespace
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Warning: Error loading checkpoint with weights_only=False: {e}")
        print("Trying alternative loading method...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

def predict_image(image_path, model, transform, classes, device='cuda', top_k=5):
    """Predict the class of a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top-k predictions
    top_prob, top_indices = torch.topk(probabilities, top_k)
    
    results = []
    for i in range(top_k):
        class_idx = top_indices[i].item()
        prob = top_prob[i].item()
        class_name = classes[class_idx]
        results.append({
            'class': class_name,
            'probability': prob,
            'confidence': prob * 100
        })
    
    return results

def plot_confusion_matrix(y_true, y_pred, classes, output_path='confusion_matrix.png', figsize=(24, 22)):
    """
    Plot and save confusion matrix in high quality
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Create figure with high DPI
    plt.figure(figsize=figsize, dpi=150)
    
    # Plot heatmap with numbers only (including zeros)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray',
                annot_kws={'fontsize': 10})
    
    plt.title('Confusion Matrix', fontsize=20, pad=20, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=16, fontweight='bold')
    plt.xlabel('Prediction', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    
    # Save figure in high quality
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Confusion matrix saved to: {output_path}")
    
    plt.close()
    
    return cm

def test_on_csv(csv_path, image_base_path, model, transform, classes, device='cuda', 
                output_csv='test_predictions.csv', save_confusion_matrix=False, 
                confusion_matrix_path='confusion_matrix.png',
                classification_report_path='classification_report.txt'):
    """Test model on all images in test.csv and save predictions"""
    
    # Read test.csv
    print(f"Reading test CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} images to test")
    
    # Lists to store results
    image_paths = []
    ground_truths = []
    predictions = []
    confidences = []
    is_correct = []
    calories_list = []
    
    # Process each image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Testing images"):
        # Get image path
        rel_image_path = row['Image_Path']
        full_image_path = os.path.join(image_base_path, rel_image_path)
        
        # Get ground truth
        ground_truth = row['Food_Label']
        calories = row['Calories']
        
        # Check if image exists
        if not os.path.exists(full_image_path):
            print(f"\nWarning: Image not found: {full_image_path}")
            image_paths.append(rel_image_path)
            ground_truths.append(ground_truth)
            predictions.append("IMAGE_NOT_FOUND")
            confidences.append(0.0)
            is_correct.append(False)
            calories_list.append(calories)
            continue
        
        try:
            # Get prediction
            results = predict_image(full_image_path, model, transform, classes, device, top_k=1)
            
            predicted_class = results[0]['class']
            confidence = results[0]['confidence']
            
            # Store results
            image_paths.append(rel_image_path)
            ground_truths.append(ground_truth)
            predictions.append(predicted_class)
            confidences.append(confidence)
            is_correct.append(predicted_class == ground_truth)
            calories_list.append(calories)
            
        except Exception as e:
            print(f"\nError processing {full_image_path}: {e}")
            image_paths.append(rel_image_path)
            ground_truths.append(ground_truth)
            predictions.append("ERROR")
            confidences.append(0.0)
            is_correct.append(False)
            calories_list.append(calories)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Image_Path': image_paths,
        'Ground_Truth': ground_truths,
        'Prediction': predictions,
        'Confidence': confidences,
        'Correct': is_correct,
        'Calories': calories_list
    })
    
    # Calculate and display accuracy
    total = len(results_df)
    correct = results_df['Correct'].sum()
    accuracy = (correct / total) * 100
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")
    
    print(f"\nTest Results Summary:")
    print(f"{'='*60}")
    print(f"Total images tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Generate confusion matrix and classification report if requested
    if save_confusion_matrix:
        print("\nGenerating confusion matrix and classification report...")
        
        # Filter out errors and missing images for metrics
        valid_df = results_df[~results_df['Prediction'].isin(['ERROR', 'IMAGE_NOT_FOUND'])]
        
        if len(valid_df) > 0:
            y_true = valid_df['Ground_Truth'].tolist()
            y_pred = valid_df['Prediction'].tolist()
            
            # Plot confusion matrix
            cm = plot_confusion_matrix(y_true, y_pred, classes, confusion_matrix_path)
            
            # Generate classification report
            report = generate_classification_report(y_true, y_pred, classes, classification_report_path)
            
            # Calculate per-class accuracy
            print(f"\n{'='*60}")
            print("PER-CLASS ACCURACY")
            print(f"{'='*60}")
            
            for i, cls in enumerate(classes):
                if cm[i].sum() > 0:
                    cls_accuracy = (cm[i, i] / cm[i].sum()) * 100
                    print(f"{cls:<30} {cls_accuracy:>6.2f}% ({cm[i, i]}/{int(cm[i].sum())})")
            print(f"{'='*60}")
        else:
            print("No valid predictions to generate confusion matrix")
    
    return results_df, accuracy

def single_image_inference(image_path, model, transform, classes, device='cuda', top_k=5):
    """Predict the class of a single image (original functionality)"""
    print(f"\nPredicting image: {image_path}")
    results = predict_image(image_path, model, transform, classes, device, top_k)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['class']:<30} {result['confidence']:>6.2f}%")
    
    print("="*60)
    print(f"\nTop prediction: {results[0]['class']} ({results[0]['confidence']:.2f}% confidence)")
    print("="*60)

def main():
    parser = argparse.ArgumentParser('Food Image Classification Inference')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single image file (for single image inference)')
    parser.add_argument('--test-csv', type=str, default=None,
                        help='Path to test.csv file (for batch testing)')
    parser.add_argument('--image-base-path', type=str, default='/content/food_dataset/images',
                        help='Base path for images in test.csv')
    parser.add_argument('--output-csv', type=str, default='test_predictions.csv',
                        help='Output CSV file path for test results')
    parser.add_argument('--save-cm', action='store_true',
                        help='Save confusion matrix and classification report')
    parser.add_argument('--cm-path', type=str, default='confusion_matrix.png',
                        help='Output path for confusion matrix image')
    parser.add_argument('--report-path', type=str, default='classification_report.txt',
                        help='Output path for classification report')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--model', type=str, default='nextvit_small',
                        help='Model name')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Show top-k predictions (for single image mode)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if either image or test-csv is provided
    if args.image is None and args.test_csv is None:
        parser.error("Either --image or --test-csv must be provided")
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Get classes
    classes = get_food_classes()
    num_classes = len(classes)
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model, num_classes, args.device)
    
    print(f"Creating transform with input size {args.input_size}...")
    transform = get_transform(args.input_size)
    
    # Test on CSV or single image
    if args.test_csv:
        print(f"\nRunning batch testing on: {args.test_csv}")
        test_on_csv(args.test_csv, args.image_base_path, model, transform, classes, 
                   args.device, args.output_csv, args.save_cm, args.cm_path, args.report_path)
    else:
        # Single image inference
        single_image_inference(args.image, model, transform, classes, args.device, args.top_k)

if __name__ == '__main__':
    main()
