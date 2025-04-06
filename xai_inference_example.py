import os
import torch
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import sys
import torch.nn as nn

# Import XAI functions
sys.path.append(os.getcwd())
from explainable_ai import (
    preprocess_image,
    generate_multi_technique_comparison,
    generate_class_activation_maps
)

# Constants from the original training file (improved_classification.py)
IMAGE_SIZE = 299  # 299x299 as specified in the Config class
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
MODEL_TYPE = "efficientnet"  # Default model type used in training

def load_class_mapping(mapping_path):
    """Load class mapping from a JSON file."""
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Handle different formats of class mapping
    if isinstance(class_mapping, dict) and all(k.isdigit() for k in class_mapping.keys()):
        # Format: {"0": "class1", "1": "class2", ...}
        idx_to_class = {int(k): v for k, v in class_mapping.items()}
        class_to_idx = {v: int(k) for k, v in class_mapping.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    elif "class_names" in class_mapping and "class_to_idx" in class_mapping:
        # Format: {"class_names": [...], "class_to_idx": {...}}
        class_names = class_mapping["class_names"]
        class_to_idx = class_mapping["class_to_idx"]
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    else:
        raise ValueError("Unsupported class mapping format")
        
    return class_names, class_to_idx, idx_to_class

def load_model(model_path, num_classes, device):
    """
    Load the trained model with EfficientNet-B3 architecture,
    exactly as defined in the improved_classification.py file.
    """
    print("Using EfficientNet-B3 model as specified in training")
    from torchvision.models import efficientnet_b3
    
    # Create model with the same architecture as in training
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle potential mismatch in classifier naming
    if "classifier.weight" in state_dict and "classifier.1.weight" not in state_dict:
        state_dict["classifier.1.weight"] = state_dict.pop("classifier.weight")
        state_dict["classifier.1.bias"] = state_dict.pop("classifier.bias")
    
    # Load model weights with non-strict matching to handle version differences
    model.load_state_dict(state_dict, strict=False)
    print(f"Successfully loaded EfficientNet-B3 model with {sum(p.numel() for p in model.parameters())} parameters")
    
    model = model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='XAI Inference Example')
    parser.add_argument('--image_path', required=True, help='Path to the image for inference')
    parser.add_argument('--model_path', default='wildlife_classifier_best.pth', help='Path to the trained model')
    parser.add_argument('--class_mapping', default='class_mapping.json', help='Path to class mapping JSON')
    parser.add_argument('--output_dir', default='xai_inference_example', help='Directory to save results')
    parser.add_argument('--skip_occlusion', action='store_true', help='Skip the slow occlusion method')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class mapping
    class_names, class_to_idx, idx_to_class = load_class_mapping(args.class_mapping)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Load model
    model = load_model(args.model_path, num_classes, device)
    print(f"Loaded model from {args.model_path}")
    
    # Create transform for inference - using the exact same transforms as in training
    inference_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    
    # Load and transform image for inference
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    # Get class name and confidence
    class_idx = predicted_idx.item()
    class_name = idx_to_class[class_idx]
    confidence_value = confidence.item()
    
    # Print prediction result
    print(f"\nPrediction:")
    print(f"Class: {class_name}")
    print(f"Confidence: {confidence_value:.4f}")
    
    # Get top-5 predictions
    top5_values, top5_indices = torch.topk(probabilities, min(5, len(class_names)))
    print("\nTop 5 predictions:")
    for i, (idx, val) in enumerate(zip(top5_indices, top5_values), 1):
        cls = idx_to_class[idx.item()]
        print(f"  {i}. {cls}: {val.item():.4f}")
    
    # Save original image with prediction
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(f"Predicted: {class_name} ({confidence_value:.2f})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'prediction.png'))
    plt.close()
    
    # Get preprocessed image for XAI
    img_tensor, img = preprocess_image(args.image_path, IMAGE_SIZE)
    
    # Generate attribution visualizations
    print("\nGenerating attribution visualizations...")
    
    # Generate multi-technique comparison
    generate_multi_technique_comparison(
        model, 
        img_tensor.to(device), 
        img, 
        class_idx, 
        class_name,
        os.path.join(args.output_dir, 'multi_technique_comparison.png'),
        skip_occlusion=args.skip_occlusion
    )
    
    # Generate class activation map
    generate_class_activation_maps(
        model,
        img_tensor.to(device),
        img,
        class_idx,
        class_name,
        os.path.join(args.output_dir, 'class_activation_map.png')
    )
    
    print(f"\nResults saved to {args.output_dir}/")
    print(f"\nModel details:")
    print(f"Architecture: EfficientNet-B3 (as used in training)")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
    print(f"Normalization: mean={NORMALIZE_MEAN}, std={NORMALIZE_STD}")

if __name__ == "__main__":
    main() 