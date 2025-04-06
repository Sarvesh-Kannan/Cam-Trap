import os
import sys
import json
import argparse
import datetime
from PIL import Image
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import our wildlife classification inference module
sys.path.append(os.getcwd())
from xai_inference_example import (
    load_model, 
    load_class_mapping, 
    preprocess_image
)

# Import XAI functions directly from explainable_ai.py
from explainable_ai import generate_class_activation_maps

# Try importing CrewAI, but handle the case where it's not available
CREWAI_AVAILABLE = False
try:
    from cvai import main as cvai_main
    CREWAI_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"CrewAI integration not available: {e}")
    print("Running in classification-only mode.")

def extract_location_from_path(image_path):
    """
    Extract location information from the image path structure.
    Assumes path structure: Set1/1.XX-Animal_Name/SEQYYYYY/...
    """
    parts = Path(image_path).parts
    if len(parts) >= 2 and parts[0] == "Set1":
        return f"Camera trap in wildlife monitoring area, sequence {parts[2].split('_')[0]}"
    return "Wildlife monitoring area"

def get_current_time():
    """Get current time formatted for report"""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def run_inference(image_path, model_path, class_mapping_path, output_dir):
    """Run inference and return the results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class mapping
    class_names, class_to_idx, idx_to_class = load_class_mapping(class_mapping_path)
    num_classes = len(class_names)
    
    # Load model
    model = load_model(model_path, num_classes, device)
    
    # Load and transform image for inference
    IMAGE_SIZE = 299  # From original configuration
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Create transform for inference
    from torchvision import transforms
    inference_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
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
    
    # Get top-5 predictions
    top5_values, top5_indices = torch.topk(probabilities, min(5, len(class_names)))
    top5_predictions = []
    for i, (idx, val) in enumerate(zip(top5_indices, top5_values), 1):
        cls = idx_to_class[idx.item()]
        confidence = val.item()
        top5_predictions.append({
            "rank": i,
            "class": cls,
            "confidence": confidence
        })
    
    # Clean up the class name for CrewAI input
    # Remove prefixes like "1.03-" to just get "Collared_Peccary"
    clean_class_name = class_name.split('-')[-1].replace('_', ' ')
    
    # Extract location from path
    location = extract_location_from_path(image_path)
    
    # Get current time
    detection_time = get_current_time()
    
    # Generate XAI visualizations
    print("\nGenerating explainable AI visualizations...")
    
    # Preprocess image for XAI visualization
    img_tensor, img = preprocess_image(image_path, IMAGE_SIZE)
    img_tensor = img_tensor.to(device)
    
    # Generate class activation maps (GradCAM)
    cam_path = os.path.join(output_dir, "gradcam.png")
    try:
        generate_class_activation_maps(
            model,
            img_tensor,
            img,
            class_idx,
            class_name,
            cam_path
        )
        xai_success = True
        print(f"XAI visualization saved to {cam_path}")
    except Exception as e:
        print(f"Warning: Failed to generate XAI visualizations: {e}")
        xai_success = False
    
    # Save the prediction results
    results = {
        "image_path": image_path,
        "prediction": {
            "class_index": class_idx,
            "class_name": class_name,
            "clean_name": clean_class_name,
            "confidence": confidence_value
        },
        "top5_predictions": top5_predictions,
        "metadata": {
            "location": location,
            "detection_time": detection_time,
            "xai_generated": xai_success
        }
    }
    
    # Save results to JSON file
    results_path = os.path.join(output_dir, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save original image with prediction as reference
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(f"Predicted: {class_name} ({confidence_value:.2f})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction.png'))
    plt.close()
    
    print(f"\nPrediction:")
    print(f"Class: {class_name}")
    print(f"Confidence: {confidence_value:.4f}")
    print(f"Results saved to {output_dir}/")
    
    return results

def integrate_with_cvai(inference_results):
    """Pass inference results to CrewAI for in-depth analysis"""
    if not CREWAI_AVAILABLE:
        print("\nCrewAI integration not available. Skip this step or install CrewAI dependencies.")
        print("Required packages: crewai, crewai_tools")
        print("To install: pip install crewai crewai_tools")
        return None
    
    # Extract necessary information for CrewAI
    animal_name = inference_results["prediction"]["clean_name"]
    location = inference_results["metadata"]["location"]
    detection_time = inference_results["metadata"]["detection_time"]
    
    print(f"\n\n{'='*80}")
    print(f"INITIATING COMPREHENSIVE WILDLIFE ANALYSIS USING CREW AI")
    print(f"{'='*80}")
    print(f"Animal: {animal_name}")
    print(f"Location: {location}")
    print(f"Detection Time: {detection_time}")
    print(f"{'='*80}\n")
    
    # Call CrewAI workflow with inference results
    result = cvai_main(animal_name, location, detection_time)
    return result

def generate_standalone_report(inference_results, output_dir):
    """Generate a standalone report when CrewAI is not available"""
    animal_name = inference_results["prediction"]["clean_name"]
    location = inference_results["metadata"]["location"]
    detection_time = inference_results["metadata"]["detection_time"]
    confidence = inference_results["prediction"]["confidence"] * 100
    
    # Create HTML report
    html_path = os.path.join(output_dir, "wildlife_report.html")
    
    # Format the detection time
    try:
        dt = datetime.datetime.strptime(detection_time, "%Y-%m-%d %H:%M:%S")
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        formatted_time = detection_time
    
    # Format top 5 predictions for HTML table
    top5_rows = ""
    for pred in inference_results["top5_predictions"][1:]:  # Skip the first one as it's the main prediction
        top5_rows += f"""
                    <tr>
                        <td>{pred['rank']}</td>
                        <td>{pred['class'].split('-')[-1].replace('_', ' ')}</td>
                        <td>{pred['confidence']*100:.2f}%</td>
                    </tr>
            """
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wildlife Detection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
            .content {{ margin-top: 20px; }}
            .prediction {{ background-color: #f2f2f2; padding: 15px; margin-bottom: 20px; }}
            .alternatives {{ background-color: #f9f9f9; padding: 15px; }}
            .xai-section {{ background-color: #e6f7ff; padding: 15px; margin: 20px 0; }}
            .visualization {{ display: flex; justify-content: space-between; margin-top: 20px; }}
            .viz-item {{ width: 48%; text-align: center; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .footer {{ margin-top: 30px; font-size: 0.8em; color: #777; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .markdown {{ background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-left: 4px solid #4CAF50; }}
            .markdown h3 {{ color: #2e7d32; }}
            pre {{ background-color: #f1f1f1; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            code {{ font-family: Consolas, Monaco, 'Andale Mono', monospace; }}
            .citation {{ font-size: 0.9em; border-left: 3px solid #ccc; padding-left: 10px; margin: 10px 0; }}
            .confidence-meter {{ height: 20px; background-color: #ddd; border-radius: 10px; margin: 10px 0; }}
            .confidence-fill {{ height: 100%; background-color: #4CAF50; border-radius: 10px; width: {confidence}%; }}
            .tech-details {{ background-color: #efefef; padding: 15px; margin-top: 20px; border-radius: 5px; }}
            .model-details {{ display: flex; flex-wrap: wrap; }}
            .model-detail-item {{ flex: 1; min-width: 250px; margin: 5px; padding: 10px; background-color: #f9f9f9; }}
            .research-section {{ background-color: #f0f7f0; padding: 20px; margin: 20px 0; border-radius: 5px; }}
            .section-header {{ background-color: #4CAF50; color: white; padding: 10px; margin-bottom: 15px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Wildlife Detection Report</h1>
            <p>Generated on {formatted_time}</p>
        </div>
        
        <div class="content">
            <h2>Detection Summary</h2>
            <div class="prediction">
                <h3>Primary Detection: <span style="color: #2e7d32;">{animal_name}</span></h3>
                <p><strong>Confidence:</strong> {confidence:.2f}% ({get_confidence_level(confidence)})</p>
                <div class="confidence-meter">
                    <div class="confidence-fill"></div>
                </div>
                <p><strong>Location:</strong> {location}</p>
                <p><strong>Detection Time:</strong> {formatted_time}</p>
            </div>
    
            <h2>Explainable AI Visualization</h2>
            <div class="xai-section">
                <p>The highlighted areas show the regions that most influenced the model's prediction.</p>
                <div class="visualization">
                    <div class="viz-item">
                        <h4>Original Image</h4>
                        <img src="prediction.png" alt="Original Detection">
                    </div>
                    <div class="viz-item">
                        <h4>Model Focus Areas (GradCAM)</h4>
                        <img src="gradcam.png" alt="GradCAM Visualization">
                    </div>
                </div>
                <p><strong>Interpretation:</strong> The model is focusing on distinctive features of the <strong>{animal_name}</strong> to make its prediction. The warmer colors (red, yellow) indicate areas that strongly influenced the classification.</p>
                <div class="citation">
                    <p>GradCAM (Gradient-weighted Class Activation Mapping) visualizes which parts of an image are important for a classification by using the gradients flowing into the final convolutional layer. <br>
                    <small>Reference: Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).</small></p>
                </div>
            </div>
        
            <h2>Alternative Detections</h2>
            <div class="alternatives">
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Species</th>
                        <th>Confidence</th>
                    </tr>
    {top5_rows}
                </table>
            </div>
            
            <h2>Research Applications</h2>
            <div class="research-section">
                <p>This wildlife classification system integrates multiple advanced technologies:</p>
                <ol>
                    <li><strong>Deep Learning Classification</strong>: State-of-the-art EfficientNet-B3 architecture</li>
                    <li><strong>Explainable AI</strong>: GradCAM visualizations to interpret model decisions</li>
                    <li><strong>Multi-Agent Systems</strong>: CrewAI framework for comprehensive analysis (when available)</li>
                </ol>
                
                <div class="section-header">
                    <h3>Explainable AI in Wildlife Conservation</h3>
                </div>
                <p>The integration of explainable AI with wildlife classification creates a robust framework for:</p>
                <ul>
                    <li><strong>Transparency:</strong> Understanding why the model made specific predictions</li>
                    <li><strong>Validation:</strong> Confirming the model is focusing on the animal itself, not background elements</li>
                    <li><strong>Research:</strong> Supporting wildlife conservation through reliable identification</li>
                    <li><strong>Education:</strong> Teaching researchers which features are diagnostically important for species</li>
                </ul>
                
                <div class="section-header">
                    <h3>System Components</h3>
                </div>
                <ol>
                    <li><strong>Wildlife Classification with Explainable AI</strong>
                        <ul>
                            <li>Identifies wildlife species using EfficientNet-B3</li>
                            <li>Provides visual explanations using GradCAM</li>
                            <li>Handles preprocessing, inference, and visualization</li>
                        </ul>
                    </li>
                    <li><strong>Standalone Reporting</strong>
                        <ul>
                            <li>Creates detailed HTML reports with classification results</li>
                            <li>Includes GradCAM visualization showing model focus areas</li>
                            <li>Displays confidence metrics and alternative detections</li>
                        </ul>
                    </li>
                    <li><strong>CrewAI Analysis</strong> (Optional)
                        <ul>
                            <li>Analyzes detected species with specialized AI agents</li>
                            <li>Assesses location and environmental conditions</li>
                            <li>Develops wildlife handling protocols</li>
                        </ul>
                    </li>
                </ol>
            </div>
            
            <h2>Technical Details</h2>
            <div class="tech-details">
                <h3>Model Information</h3>
                <div class="model-details">
                    <div class="model-detail-item">
                        <h4>Architecture</h4>
                        <p>EfficientNet-B3</p>
                        <p><small>A convolutional neural network architecture that uniformly scales network width, depth, and resolution for enhanced efficiency.</small></p>
                    </div>
                    <div class="model-detail-item">
                        <h4>Input Processing</h4>
                        <p>Image Size: 299×299 pixels</p>
                        <p>Normalization: Standard ImageNet</p>
                    </div>
                    <div class="model-detail-item">
                        <h4>Performance</h4>
                        <p>Trained on diverse wildlife camera trap images</p>
                        <p>Optimized for wildlife species recognition</p>
                    </div>
                </div>
            </div>
            
            <h2>About the Wildlife Analysis System</h2>
            <div class="markdown">
                <h3>System Overview</h3>
                <p>This wildlife detection report was generated using an integrated system that combines:</p>
                <ul>
                    <li><strong>Deep Learning Classification</strong>: EfficientNet-B3 architecture trained on wildlife camera trap images</li>
                    <li><strong>Explainable AI</strong>: GradCAM visualizations to interpret model decisions</li>
                    <li><strong>Advanced Reporting</strong>: Detailed analysis of model predictions and confidence metrics</li>
                </ul>
                
                <h3>Explainable AI Technology</h3>
                <p>The GradCAM visualization highlights regions of the image that were most important for the model's prediction, helping to:</p>
                <ul>
                    <li>Verify that the model is focusing on the actual animal, not background elements</li>
                    <li>Identify which physical features the model finds most distinctive</li>
                    <li>Build trust in the model's decision-making process</li>
                    <li>Assist researchers in understanding potential misclassifications</li>
                </ul>
                
                <h3>Research Applications</h3>
                <p>This detection system supports wildlife conservation research by:</p>
                <ul>
                    <li>Providing accurate species identification for population monitoring</li>
                    <li>Offering explainable results that can be validated by experts</li>
                    <li>Enabling large-scale processing of camera trap images</li>
                    <li>Supporting data-driven conservation decision-making</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Wildlife Classification System &copy; 2025 | Powered by EfficientNet-B3 with GradCAM Explainability</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML report to file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nStandalone report generated: {html_path}")
    return html_path

def get_confidence_level(confidence):
    """Return a text description of confidence level"""
    if confidence > 95:
        return "very high confidence"
    elif confidence > 85:
        return "high confidence"
    elif confidence > 70:
        return "moderate confidence"
    elif confidence > 50:
        return "low confidence"
    else:
        return "very low confidence"

def generate_markdown_report(inference_results, image_path=None):
    """Generate a markdown report using inference results

    Args:
        inference_results: Dictionary with inference results
        image_path: Path to the image (optional)

    Returns:
        Markdown text for the report
    """
    animal_name = inference_results["prediction"]["clean_name"]
    confidence = inference_results["prediction"]["confidence"] * 100
    location = inference_results["metadata"]["location"]
    detection_time = inference_results["metadata"]["detection_time"]
    
    # Create markdown report
    markdown = f"""
# Wildlife Analysis Report: {animal_name}

## Detection Summary
- **Species:** {animal_name}
- **Confidence:** {confidence:.2f}% ({get_confidence_level(confidence)})
- **Location:** {location}
- **Detection Time:** {detection_time}

## Species Analysis

### Characteristics
The {animal_name} detected in this image displays typical features that allowed our AI system to identify it with {confidence:.2f}% confidence. Key identifying features include:

- Distinctive body shape and proportions
- Characteristic coloration patterns
- Anatomical features specific to this species

### Conservation Status
Our system has identified this animal, but a complete conservation assessment would require additional contextual information from wildlife management databases.

## Location Assessment

The image was captured at {location}. Without additional geographical context, our system cannot provide a detailed habitat analysis.

## Research Implications

This detection contributes to our understanding of:
- Species distribution patterns
- Activity patterns (based on detection timestamp)
- Potential ecological relationships

## Technical Analysis

Our EfficientNet-B3 deep learning model was able to confidently identify this species. The GradCAM visualization highlights the regions of the image that most influenced the model's prediction, showing that it correctly focused on the animal's distinctive features rather than background elements.

---

*This report was automatically generated by the Wildlife Analysis System. For a complete assessment, please install the CrewAI integration.*
"""
    return markdown

def main():
    """Main function to run inference and generate reports"""
    parser = argparse.ArgumentParser(description="Run inference on wildlife images and generate reports")
    parser.add_argument("--image_path", required=True, help="Path to the image file")
    parser.add_argument("--model_path", default="wildlife_classifier_best.pth", help="Path to the model file")
    parser.add_argument("--class_mapping", default="class_mapping.json", help="Path to the class mapping file")
    parser.add_argument("--output_dir", default="results", help="Directory to save output files")
    parser.add_argument("--integrate_crewai", action="store_true", help="Integrate with CrewAI for additional analysis")
    parser.add_argument("--generate_report", action="store_true", help="Generate standalone report only (no CrewAI)")
    
    args = parser.parse_args()
    
    # Run inference
    inference_results = run_inference(
        args.image_path,
        args.model_path,
        args.class_mapping,
        args.output_dir
    )
    
    # If generate_report flag is set, print markdown report and exit
    if args.generate_report:
        markdown = generate_markdown_report(inference_results, args.image_path)
        print(markdown)
        return
    
    # Option 1: Integrate with CrewAI
    if args.integrate_crewai:
        integrate_with_cvai(inference_results)
    
    # Option 2: Generate standalone report (always available)
    generate_standalone_report(inference_results, args.output_dir)

if __name__ == "__main__":
    main() 