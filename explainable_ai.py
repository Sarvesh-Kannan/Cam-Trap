import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import json
import argparse
from torchvision.models import efficientnet_b3
from torchvision.models import inception_v3
import timm
from tqdm import tqdm
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pandas as pd
import glob
from captum.attr import (
    GradientShap, 
    DeepLift, 
    IntegratedGradients,
    Occlusion,
    NoiseTunnel,
    LRP
)
from captum.attr import visualization as viz
import warnings
import random
import matplotlib.cm as cm
warnings.filterwarnings("ignore")

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define custom colors for visualizations
TURBO_COLORS = [(0.18995, 0.07176, 0.23217), (0.19483, 0.08339, 0.26149),
                (0.19956, 0.09498, 0.29024), (0.20415, 0.10652, 0.31844),
                (0.20860, 0.11802, 0.34607), (0.21291, 0.12947, 0.37314),
                (0.21708, 0.14087, 0.39964), (0.22111, 0.15223, 0.42558),
                (0.22500, 0.16354, 0.45096), (0.22875, 0.17481, 0.47578),
                (0.23236, 0.18603, 0.50004), (0.23582, 0.19720, 0.52373),
                (0.23915, 0.20833, 0.54686), (0.24234, 0.21941, 0.56942),
                (0.24539, 0.23044, 0.59142), (0.24830, 0.24143, 0.61286),
                (0.25107, 0.25237, 0.63374), (0.25369, 0.26327, 0.65406),
                (0.25618, 0.27412, 0.67381), (0.25853, 0.28492, 0.69299),
                (0.26074, 0.29568, 0.71162), (0.26280, 0.30639, 0.72968),
                (0.26473, 0.31706, 0.74718), (0.26652, 0.32768, 0.76412),
                (0.26816, 0.33825, 0.78050), (0.26967, 0.34878, 0.79631),
                (0.27103, 0.35926, 0.81156), (0.27226, 0.36970, 0.82624),
                (0.27334, 0.38008, 0.84037), (0.27429, 0.39043, 0.85393),
                (0.27509, 0.40072, 0.86692), (0.27576, 0.41097, 0.87936),
                (0.27628, 0.42118, 0.89123), (0.27667, 0.43134, 0.90254),
                (0.27691, 0.44145, 0.91328), (0.27701, 0.45152, 0.92347),
                (0.27698, 0.46153, 0.93309), (0.27680, 0.47151, 0.94214),
                (0.27648, 0.48144, 0.95064), (0.27603, 0.49132, 0.95857),
                (0.27543, 0.50115, 0.96594), (0.27469, 0.51094, 0.97275),
                (0.27381, 0.52069, 0.97899), (0.27273, 0.53040, 0.98461),
                (0.27106, 0.54015, 0.98930), (0.26878, 0.54995, 0.99303),
                (0.26592, 0.55979, 0.99583), (0.26252, 0.56967, 0.99773),
                (0.25862, 0.57958, 0.99876), (0.25425, 0.58950, 0.99896),
                (0.24946, 0.59943, 0.99835), (0.24427, 0.60937, 0.99697),
                (0.23874, 0.61931, 0.99485), (0.23288, 0.62923, 0.99202),
                (0.22676, 0.63913, 0.98851), (0.22039, 0.64901, 0.98436),
                (0.21382, 0.65886, 0.97959), (0.20708, 0.66866, 0.97423),
                (0.20021, 0.67842, 0.96833), (0.19326, 0.68812, 0.96190),
                (0.18625, 0.69775, 0.95498), (0.17923, 0.70732, 0.94761),
                (0.17223, 0.71680, 0.93981), (0.16529, 0.72620, 0.93161),
                (0.15844, 0.73551, 0.92305), (0.15173, 0.74472, 0.91416),
                (0.14519, 0.75381, 0.90496), (0.13886, 0.76279, 0.89550),
                (0.13278, 0.77165, 0.88580), (0.12698, 0.78037, 0.87590),
                (0.12151, 0.78896, 0.86581), (0.11639, 0.79740, 0.85559),
                (0.11167, 0.80569, 0.84525), (0.10738, 0.81381, 0.83484),
                (0.10357, 0.82177, 0.82437), (0.10026, 0.82955, 0.81389),
                (0.09750, 0.83714, 0.80342), (0.09532, 0.84455, 0.79299),
                (0.09377, 0.85175, 0.78264), (0.09287, 0.85875, 0.77240),
                (0.09267, 0.86554, 0.76230), (0.09320, 0.87211, 0.75237),
                (0.09451, 0.87844, 0.74265), (0.09662, 0.88454, 0.73316),
                (0.09958, 0.89040, 0.72393), (0.10342, 0.89600, 0.71500),
                (0.10815, 0.90142, 0.70599), (0.11374, 0.90673, 0.69651),
                (0.12014, 0.91193, 0.68660), (0.12733, 0.91701, 0.67627),
                (0.13526, 0.92197, 0.66556), (0.14391, 0.92680, 0.65448),
                (0.15323, 0.93151, 0.64308), (0.16319, 0.93609, 0.63137),
                (0.17377, 0.94053, 0.61938), (0.18491, 0.94484, 0.60713),
                (0.19659, 0.94901, 0.59466), (0.20877, 0.95304, 0.58199),
                (0.22142, 0.95692, 0.56914), (0.23449, 0.96065, 0.55614),
                (0.24797, 0.96423, 0.54303), (0.26180, 0.96765, 0.52981),
                (0.27597, 0.97092, 0.51653), (0.29042, 0.97403, 0.50321),
                (0.30513, 0.97697, 0.48987), (0.32006, 0.97974, 0.47654),
                (0.33517, 0.98234, 0.46325), (0.35043, 0.98477, 0.45002),
                (0.36581, 0.98702, 0.43688), (0.38127, 0.98909, 0.42386),
                (0.39678, 0.99098, 0.41098), (0.41229, 0.99268, 0.39826),
                (0.42778, 0.99419, 0.38575), (0.44321, 0.99551, 0.37345),
                (0.45854, 0.99663, 0.36140), (0.47375, 0.99755, 0.34963),
                (0.48879, 0.99828, 0.33816), (0.50362, 0.99879, 0.32701),
                (0.51822, 0.99910, 0.31622), (0.53255, 0.99919, 0.30581),
                (0.54658, 0.99907, 0.29581), (0.56026, 0.99873, 0.28623),
                (0.57357, 0.99817, 0.27712), (0.58646, 0.99739, 0.26849),
                (0.59891, 0.99638, 0.26038), (0.61088, 0.99514, 0.25280),
                (0.62233, 0.99366, 0.24579), (0.63323, 0.99195, 0.23937),
                (0.64362, 0.98999, 0.23356), (0.65394, 0.98775, 0.22835),
                (0.66428, 0.98524, 0.22370), (0.67462, 0.98246, 0.21960),
                (0.68494, 0.97941, 0.21602), (0.69525, 0.97610, 0.21294),
                (0.70553, 0.97255, 0.21032), (0.71577, 0.96875, 0.20815),
                (0.72596, 0.96470, 0.20640), (0.73610, 0.96043, 0.20504),
                (0.74617, 0.95593, 0.20406), (0.75617, 0.95121, 0.20343),
                (0.76608, 0.94627, 0.20311), (0.77591, 0.94113, 0.20310),
                (0.78563, 0.93579, 0.20336), (0.79524, 0.93025, 0.20386),
                (0.80473, 0.92452, 0.20459), (0.81410, 0.91861, 0.20552),
                (0.82333, 0.91253, 0.20663), (0.83241, 0.90627, 0.20788),
                (0.84133, 0.89986, 0.20926), (0.85010, 0.89328, 0.21074),
                (0.85868, 0.88655, 0.21230), (0.86709, 0.87968, 0.21391),
                (0.87530, 0.87267, 0.21555), (0.88331, 0.86553, 0.21719),
                (0.89112, 0.85826, 0.21880), (0.89870, 0.85087, 0.22038),
                (0.90605, 0.84337, 0.22188), (0.91317, 0.83576, 0.22328),
                (0.92004, 0.82806, 0.22456), (0.92666, 0.82025, 0.22570),
                (0.93301, 0.81236, 0.22667), (0.93909, 0.80439, 0.22744),
                (0.94489, 0.79634, 0.22800), (0.95039, 0.78823, 0.22831),
                (0.95560, 0.78005, 0.22836), (0.96049, 0.77181, 0.22811),
                (0.96507, 0.76352, 0.22754), (0.96931, 0.75519, 0.22663),
                (0.97323, 0.74682, 0.22536), (0.97679, 0.73842, 0.22369),
                (0.98000, 0.73000, 0.22161), (0.98289, 0.72140, 0.21918),
                (0.98549, 0.71250, 0.21650), (0.98781, 0.70330, 0.21358),
                (0.98986, 0.69382, 0.21043), (0.99163, 0.68408, 0.20706),
                (0.99314, 0.67408, 0.20348), (0.99438, 0.66386, 0.19971),
                (0.99535, 0.65341, 0.19577), (0.99607, 0.64277, 0.19165),
                (0.99654, 0.63193, 0.18738), (0.99675, 0.62093, 0.18297),
                (0.99672, 0.60977, 0.17842), (0.99644, 0.59846, 0.17376),
                (0.99593, 0.58703, 0.16899), (0.99517, 0.57549, 0.16412),
                (0.99419, 0.56386, 0.15918), (0.99297, 0.55214, 0.15417),
                (0.99153, 0.54036, 0.14910), (0.98987, 0.52854, 0.14398),
                (0.98799, 0.51667, 0.13883), (0.98590, 0.50479, 0.13367),
                (0.98360, 0.49291, 0.12849), (0.98108, 0.48104, 0.12332),
                (0.97837, 0.46920, 0.11817), (0.97545, 0.45740, 0.11305),
                (0.97234, 0.44565, 0.10797), (0.96904, 0.43399, 0.10294),
                (0.96555, 0.42241, 0.09798), (0.96187, 0.41093, 0.09310),
                (0.95801, 0.39958, 0.08831), (0.95398, 0.38836, 0.08362),
                (0.94977, 0.37729, 0.07905), (0.94538, 0.36638, 0.07461),
                (0.94084, 0.35566, 0.07031), (0.93612, 0.34513, 0.06616),
                (0.93125, 0.33482, 0.06218), (0.92623, 0.32473, 0.05837),
                (0.92105, 0.31489, 0.05475), (0.91572, 0.30530, 0.05134),
                (0.91024, 0.29599, 0.04814), (0.90463, 0.28696, 0.04516),
                (0.89888, 0.27824, 0.04243), (0.89298, 0.26981, 0.03993),
                (0.88691, 0.26152, 0.03753), (0.88066, 0.25334, 0.03521),
                (0.87422, 0.24526, 0.03297), (0.86760, 0.23730, 0.03082),
                (0.86079, 0.22945, 0.02875), (0.85380, 0.22170, 0.02677),
                (0.84662, 0.21407, 0.02487), (0.83926, 0.20654, 0.02305),
                (0.83172, 0.19912, 0.02131), (0.82399, 0.19182, 0.01966),
                (0.81608, 0.18462, 0.01809), (0.80799, 0.17753, 0.01660),
                (0.79971, 0.17055, 0.01520), (0.79125, 0.16368, 0.01387),
                (0.78260, 0.15693, 0.01264), (0.77377, 0.15028, 0.01148),
                (0.76476, 0.14374, 0.01041), (0.75556, 0.13731, 0.00942),
                (0.74617, 0.13098, 0.00851), (0.73661, 0.12477, 0.00769),
                (0.72686, 0.11867, 0.00695), (0.71692, 0.11268, 0.00629),
                (0.70680, 0.10680, 0.00571), (0.69650, 0.10102, 0.00522),
                (0.68602, 0.09536, 0.00481), (0.67535, 0.08980, 0.00449),
                (0.66449, 0.08436, 0.00424), (0.65345, 0.07902, 0.00408),
                (0.64223, 0.07380, 0.00401), (0.63082, 0.06868, 0.00401),
                (0.61923, 0.06367, 0.00410), (0.60746, 0.05878, 0.00427),
                (0.59550, 0.05399, 0.00453), (0.58336, 0.04931, 0.00486),
                (0.57103, 0.04474, 0.00529), (0.55852, 0.04028, 0.00579),
                (0.54583, 0.03593, 0.00638), (0.53295, 0.03169, 0.00705),
                (0.51989, 0.02756, 0.00780), (0.50664, 0.02354, 0.00863),
                (0.49321, 0.01963, 0.00955), (0.47960, 0.01583, 0.01055)]

turbo_cmap = LinearSegmentedColormap.from_list("turbo", TURBO_COLORS)

def load_model(model_path, model_type, num_classes, device):
    """
    Load the trained model based on model_type
    """
    if model_type == "efficientnet":
        model = efficientnet_b3(weights=None)
        model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_type == "vit":
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    
    elif model_type == "inception":
        model = inception_v3(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.aux_logits = False  # Disable auxiliary output for inference
    
    elif model_type == "ensemble":
        # For ensemble model, we need a custom class similar to what we defined in training
        efficientnet = efficientnet_b3(weights=None)
        efficientnet.classifier = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
        
        vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        
        class EnsembleModel(nn.Module):
            def __init__(self, model1, model2):
                super(EnsembleModel, self).__init__()
                self.model1 = model1  # EfficientNet
                self.model2 = model2  # ViT
                self.classifier = nn.Linear(num_classes * 2, num_classes)
                
            def forward(self, x):
                # For ViT, resize to 224x224
                x_vit = torch.nn.functional.interpolate(
                    x, size=(224, 224), mode='bilinear', align_corners=False
                )
                
                # Get features from both models
                out1 = self.model1(x)
                out2 = self.model2(x_vit)
                
                # Concatenate and pass through classifier
                combined = torch.cat((out1, out2), dim=1)
                return self.classifier(combined)
        
        model = EnsembleModel(efficientnet, vit)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path, image_size):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img

def get_attribution_method(method_name, model):
    """Create attribution method based on name"""
    if method_name == 'integrated_gradients':
        return IntegratedGradients(model)
    elif method_name == 'gradient_shap':
        return GradientShap(model)
    elif method_name == 'deeplift':
        return DeepLift(model)
    elif method_name == 'occlusion':
        return Occlusion(model)
    elif method_name == 'lrp':
        return LRP(model)
    else:
        return IntegratedGradients(model)  # Default

def normalize_attr(attr):
    """Normalize attribution tensor for visualization."""
    # Check if it's a PyTorch tensor and convert to numpy if it is
    if isinstance(attr, torch.Tensor):
        attr = attr.squeeze().cpu().detach().numpy()
        # If it's a 3D tensor with channels first, transpose to channels last
        if len(attr.shape) == 3 and attr.shape[0] <= 3:
            attr = attr.transpose(1, 2, 0)
    else:
        # Handle numpy array
        attr = attr.squeeze()
    
    # For single channel attributions, convert to 3-channel for visualization
    if len(attr.shape) == 2:  # If 2D (single channel)
        attr = np.expand_dims(attr, axis=2)
        attr = np.repeat(attr, 3, axis=2)
    elif len(attr.shape) == 3 and attr.shape[2] == 1:  # If already 3D but single channel
        attr = np.repeat(attr, 3, axis=2)
    
    # Print attribution statistics for debugging
    print(f"Attribution min: {attr.min()}, max: {attr.max()}, mean: {attr.mean()}")
    
    # Use percentile-based normalization to highlight subtle features
    # Get 1st and 99th percentiles to avoid outliers
    p_low, p_high = np.percentile(attr, [1, 99])
    # Expand the range slightly to ensure some contrast
    attr_range = max(p_high - p_low, 1e-6)
    
    # Normalize to [0, 1] range based on percentiles
    attr = np.clip((attr - p_low) / attr_range, 0, 1)
    
    return attr

def plot_heatmap(original_img, attr, title, ax=None, cmap='turbo', alpha=0.4):
    """Plot attribution heatmap overlaid on original image"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert PIL image to numpy if needed
    if isinstance(original_img, Image.Image):
        img_np = np.array(original_img)
    else:
        img_np = original_img
    
    # Plot original image
    ax.imshow(img_np)
    
    # Ensure attr is the right shape and normalize
    # Check if it's a tensor and convert appropriately
    if isinstance(attr, torch.Tensor):
        attr_np = attr.squeeze().cpu().detach().numpy()
        if len(attr_np.shape) == 3 and attr_np.shape[0] <= 3:  # If CHW format
            # Sum across channels to get a single heatmap
            attr_np = np.sum(np.abs(attr_np), axis=0)
    else:
        attr_np = attr.squeeze()
        if len(attr_np.shape) == 3 and attr_np.shape[2] <= 3:  # If HWC format
            # Sum across channels to get a single heatmap
            attr_np = np.sum(np.abs(attr_np), axis=2)
    
    # Normalize for visualization
    attr_min, attr_max = attr_np.min(), attr_np.max()
    if attr_max > attr_min:
        attr_np = (attr_np - attr_min) / (attr_max - attr_min)
    
    # Resize to match image dimensions if needed
    if attr_np.shape != img_np.shape[:2]:
        attr_np = cv2.resize(attr_np, (img_np.shape[1], img_np.shape[0]))
    
    # Plot heatmap
    heatmap = ax.imshow(attr_np, cmap=cmap, alpha=alpha)
    ax.set_title(title)
    ax.axis('off')
    
    return heatmap

def generate_confusion_matrix(model, data_loader, device, class_names):
    """Generate a confusion matrix for the model on the test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating test data"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_detailed.png', dpi=300)
    plt.close()
    
    # Normalize confusion matrix for better visualization
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png', dpi=300)
    plt.close()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    return cm, report 

def generate_attribution_visualization(model, img_tensor, img, class_idx, class_name, method_name, output_path):
    """Generate attribution visualization using specified method"""
    # Create attribution method
    attribution_method = get_attribution_method(method_name, model)
    
    # Register hooks for LRP if selected
    if method_name == 'lrp':
        from captum.attr._utils.lrp_rules import EpsilonRule
        # Register hooks for various layer types
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.rule = EpsilonRule()
    
    # Get attributions
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad = True
    
    if method_name == 'occlusion':
        # Occlusion needs different parameters
        attributions = attribution_method.attribute(
            img_tensor,
            target=class_idx,
            sliding_window_shapes=(3, 15, 15),  # (channels, height, width)
            strides=(3, 8, 8),
            baselines=0
        )
    else:
        # Standard attribution for other methods
        attributions = attribution_method.attribute(img_tensor, target=class_idx)
    
    # Create figure for visualization
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot original image
    axs[0].imshow(img)
    axs[0].set_title(f"Original Image: {class_name}")
    axs[0].axis('off')
    
    # Plot attribution
    plot_heatmap(img, attributions, f"{method_name.replace('_', ' ').title()} Attribution", ax=axs[1])
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return attributions

def generate_multi_technique_comparison(model, img_tensor, img, class_idx, class_name, output_path, skip_occlusion=False):
    """Compare multiple attribution techniques for the same image."""
    print(f"\nGenerating multi-technique comparison for class {class_name}...")
    
    # Get the true class from the output path
    path_parts = output_path.split('/')
    file_dir = '/'.join(path_parts[:-2])  # Get the directory containing the image
    true_class = None
    for part in file_dir.split('/'):
        if part.startswith('1.'):  # Classes start with 1.XX
            true_class = part
            break
    
    # Move tensor to the same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Create a single figure rather than multiple subplots
    plt.figure(figsize=(16, 10))
    
    # Original image - top left
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    # Print misclassification warning at the top
    if true_class and true_class != class_name:
        plt.figtext(0.5, 0.95, f'MISCLASSIFICATION: True:{true_class}, Predicted:{class_name}', 
                  fontsize=16, color='red', ha='center', weight='bold')
    else:
        plt.figtext(0.5, 0.95, f'Attribution Methods for: {class_name}', 
                  fontsize=16, ha='center', weight='bold')
    
    # Integrated Gradients - top middle
    print("Computing Integrated Gradients...")
    try:
        ig = IntegratedGradients(model)
        attributions = ig.attribute(img_tensor, target=class_idx, n_steps=50)
        print(f"IG attribution shape: {attributions.shape}, min: {attributions.min().item()}, max: {attributions.max().item()}")
        
        plt.subplot(2, 3, 2)
        attr_data = normalize_attr_for_display(attributions.squeeze().cpu().detach().numpy())
        plt.imshow(attr_data, cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Integrated Gradients', fontsize=12)
        plt.axis('off')
    except Exception as e:
        print(f"Error in Integrated Gradients: {e}")
        plt.subplot(2, 3, 2)
        plt.text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
               ha='center', va='center')
        plt.axis('off')
    
    # GradientShap - top right
    print("Computing GradientShap...")
    try:
        gradient_shap = GradientShap(model)
        # Create baseline
        baseline = torch.zeros_like(img_tensor)
        attributions = gradient_shap.attribute(img_tensor, target=class_idx, baselines=baseline)
        print(f"GradientShap attribution shape: {attributions.shape}, min: {attributions.min().item()}, max: {attributions.max().item()}")
        
        plt.subplot(2, 3, 3)
        attr_data = normalize_attr_for_display(attributions.squeeze().cpu().detach().numpy())
        plt.imshow(attr_data, cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Gradient SHAP', fontsize=12)
        plt.axis('off')
    except Exception as e:
        print(f"Error in GradientShap: {e}")
        plt.subplot(2, 3, 3)
        plt.text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
               ha='center', va='center')
        plt.axis('off')
    
    # DeepLift - bottom left
    print("Computing DeepLift...")
    try:
        deeplift = DeepLift(model)
        attributions = deeplift.attribute(img_tensor, target=class_idx)
        print(f"DeepLift attribution shape: {attributions.shape}, min: {attributions.min().item()}, max: {attributions.max().item()}")
        
        plt.subplot(2, 3, 4)
        attr_data = normalize_attr_for_display(attributions.squeeze().cpu().detach().numpy())
        plt.imshow(attr_data, cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('DeepLift', fontsize=12)
        plt.axis('off')
    except Exception as e:
        print(f"Error in DeepLift: {e}")
        plt.subplot(2, 3, 4)
        plt.text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
               ha='center', va='center')
        plt.axis('off')
    
    # Occlusion - bottom middle
    if not skip_occlusion:
        print("Computing Occlusion (this may take a while)...")
        try:
            occlusion = Occlusion(model)
            attributions = occlusion.attribute(
                img_tensor, 
                target=class_idx,
                strides=(3, 4, 4),
                sliding_window_shapes=(3, 8, 8),
                baselines=0
            )
            print(f"Occlusion attribution shape: {attributions.shape}, min: {attributions.min().item()}, max: {attributions.max().item()}")
            
            plt.subplot(2, 3, 5)
            attr_data = normalize_attr_for_display(attributions.squeeze().cpu().detach().numpy())
            plt.imshow(attr_data, cmap='viridis')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('Occlusion', fontsize=12)
            plt.axis('off')
        except Exception as e:
            print(f"Error in Occlusion: {e}")
            plt.subplot(2, 3, 5)
            plt.text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                   ha='center', va='center')
            plt.axis('off')
    else:
        print("Skipping Occlusion calculation (as requested)...")
        plt.subplot(2, 3, 5)
        plt.text(0.5, 0.5, "Occlusion skipped\n(to save computation time)", 
               ha='center', va='center')
        plt.axis('off')
    
    # Add a text summary - bottom right
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create a textbox with info
    textstr = 'Explainability Summary:\n\n'
    textstr += f'Predicted: {class_name}\n'
    
    if true_class and true_class != class_name:
        textstr += f'True class: {true_class}\n'
        textstr += 'MISCLASSIFICATION DETECTED!\n\n'
    else:
        textstr += '\n'
    
    textstr += 'This visualization shows how the model\n'
    textstr += 'makes its prediction by highlighting\n'
    textstr += 'the most important regions in the image.\n\n'
    
    textstr += 'Brighter areas indicate features that\n'
    textstr += 'strongly influenced the prediction.'
    
    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.5, 0.5, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved multi-technique comparison to {output_path}")
    plt.close()

def normalize_attr_for_display(attr_array):
    """Helper function to normalize attribution arrays for display."""
    # Sum across channels if multi-channel
    if len(attr_array.shape) == 3 and attr_array.shape[0] <= 3:
        # CHW format - sum across channels
        attr_array = np.sum(np.abs(attr_array), axis=0)
    elif len(attr_array.shape) == 3 and attr_array.shape[2] <= 3:
        # HWC format - sum across channels
        attr_array = np.sum(np.abs(attr_array), axis=2)
    
    # Apply percentile-based normalization to highlight subtle features
    p_low, p_high = np.percentile(attr_array, [2, 98])
    attr_range = max(p_high - p_low, 1e-6)
    attr_array = np.clip((attr_array - p_low) / attr_range, 0, 1)
    
    return attr_array

def generate_animated_attribution(model, img_tensor, img, class_idx, class_name, output_path, frames=10):
    """Generate animated attribution visualization (showing attribution build-up)"""
    # Use Integrated Gradients for animation
    attribution_method = IntegratedGradients(model)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad = True
    
    # List to store frames
    frame_images = []
    
    # Create base figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(f"Attribution for: {class_name}", fontsize=14)
    ax.axis('off')
    
    # Generate frames with increasing steps (baselines to input)
    for steps in np.linspace(5, 100, frames, dtype=int):
        attributions = attribution_method.attribute(
            img_tensor, 
            target=class_idx,
            n_steps=steps
        )
        
        # Clear previous heatmap
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Attribution for: {class_name} (Steps: {steps})", fontsize=14)
        ax.axis('off')
        
        # Add new heatmap
        plot_heatmap(img, attributions, "", ax=ax)
        
        # Save frame
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frame_images.append(frame)
    
    plt.close()
    
    # Save as gif using imageio
    try:
        import imageio
        imageio.mimsave(output_path, frame_images, fps=2)
    except ImportError:
        print("Could not generate animation. Please install imageio: pip install imageio")
        # Save last frame as static image instead
        last_frame_path = output_path.replace('.gif', '.png')
        plt.figure(figsize=(10, 10))
        plt.imshow(frame_images[-1])
        plt.axis('off')
        plt.savefig(last_frame_path, dpi=300, bbox_inches='tight')
        plt.close()

def generate_class_activation_maps(model, img_tensor, img, class_idx, class_name, output_path):
    """Generate Class Activation Map (CAM) visualizations.
    Implements a simple version of GradCAM for EfficientNet and similar models.
    """
    print(f"\nGenerating class activation maps for class {class_name}...")
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Register hooks to capture activations and gradients
    activations = {}
    gradients = {}
    
    # For EfficientNet, we'll look at the last convolutional layer
    target_layer_name = 'features'  # This is usually the feature extractor part
    
    # Define hook functions
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    def save_gradient(name):
        def hook(module, grad_in, grad_out):
            gradients[name] = grad_out[0]
        return hook
    
    # Try to find the target layer
    # Different models have different structures, so we need to handle them differently
    target_layer = None
    if hasattr(model, 'features'):
        # EfficientNet, MobileNet, etc.
        modules = list(model.features.children())
        # Get the last convolutional layer
        for i in range(len(modules)-1, -1, -1):
            if isinstance(modules[i], torch.nn.Conv2d):
                target_layer = modules[i]
                target_layer_name = f'features.{i}'
                break
    
    if target_layer is None:
        # Another approach for models like ResNet
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_layer_name = name
    
    if target_layer is None:
        print("Could not find a suitable convolutional layer for GradCAM")
        return
    
    print(f"Using layer {target_layer_name} for GradCAM")
    
    # Register forward and backward hooks
    handle_fwd = target_layer.register_forward_hook(save_activation(target_layer_name))
    handle_bwd = target_layer.register_full_backward_hook(save_gradient(target_layer_name))
    
    # Forward pass and compute gradients
    model.zero_grad()
    output = model(img_tensor)
    target_score = output[0, class_idx]
    target_score.backward()
    
    # Compute CAM
    activations_value = activations[target_layer_name]
    gradients_value = gradients[target_layer_name]
    
    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()
    
    # Global average pooling of gradients
    weights = torch.mean(gradients_value, dim=(2, 3), keepdim=True)
    
    # Weight activations with gradients
    cam = torch.sum(weights * activations_value, dim=1, keepdim=True)
    
    # ReLU to keep only positive contributions
    cam = torch.nn.functional.relu(cam)
    
    # Normalize and convert to numpy
    cam = cam.squeeze().cpu().detach().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    # Resize CAM to match image size
    img_np = np.array(img)
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    
    # Print stats for debugging
    print(f"CAM stats - min: {cam_resized.min()}, max: {cam_resized.max()}, mean: {cam_resized.mean()}")
    
    # Enhance contrast for better visualization
    # Apply histogram equalization to improve contrast, especially for night images
    if cam_resized.mean() < 0.1:  # Very low activation mean indicates a need for enhancement
        print("Low activation detected, enhancing contrast...")
        cam_resized = cv2.equalizeHist(np.uint8(cam_resized * 255)) / 255.0
    
    # Apply aggressive percentile-based normalization for subtle features
    p_low, p_high = np.percentile(cam_resized, [5, 95])
    cam_resized = np.clip((cam_resized - p_low) / (p_high - p_low + 1e-6), 0, 1)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Class Activation Map')
    plt.axis('off')
    
    # Plot overlay
    overlay = heatmap * 0.7 + img_np * 0.3
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.suptitle(f'GradCAM for Class: {class_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved class activation map to {output_path}")
    plt.close()

def create_misclassification_analysis(model, data_loader, device, class_names, output_dir='misclassifications'):
    """Analyze model misclassifications and visualize them with attributions"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    misclassifications = []
    
    # Get all misclassified examples
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Finding misclassifications"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified examples
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if pred != label:
                    # Get input image
                    img_tensor = inputs[i:i+1].detach().cpu()
                    
                    # Get probabilities
                    probs = torch.nn.functional.softmax(outputs[i], dim=0)
                    
                    misclassifications.append({
                        'img_tensor': img_tensor,
                        'true_label': label.item(),
                        'pred_label': pred.item(),
                        'true_class': class_names[label.item()],
                        'pred_class': class_names[pred.item()],
                        'confidence': probs[pred].item()
                    })
    
    print(f"Found {len(misclassifications)} misclassified examples")
    
    # Sort by confidence (highest first)
    misclassifications.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Analyze top misclassifications (limit to 10 to avoid generating too many visualizations)
    for i, misclass in enumerate(misclassifications[:10]):
        # Convert tensor to PIL image for visualization
        img_tensor = misclass['img_tensor']
        
        # Denormalize image for display
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_tensor_display = inv_normalize(img_tensor.squeeze())
        img_display = transforms.ToPILImage()(img_tensor_display)
        
        # Create output directory for this misclassification
        misclass_dir = os.path.join(output_dir, f"misclass_{i+1}")
        os.makedirs(misclass_dir, exist_ok=True)
        
        # Save original image with true and predicted labels
        plt.figure(figsize=(8, 8))
        plt.imshow(img_display)
        plt.title(f"True: {misclass['true_class']}, Predicted: {misclass['pred_class']} ({misclass['confidence']:.2f})")
        plt.axis('off')
        plt.savefig(os.path.join(misclass_dir, "original.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate attribution for true class
        generate_attribution_visualization(
            model, 
            img_tensor.to(device), 
            img_display, 
            misclass['true_label'], 
            misclass['true_class'], 
            'integrated_gradients', 
            os.path.join(misclass_dir, "attribution_true_class.png")
        )
        
        # Generate attribution for predicted (wrong) class
        generate_attribution_visualization(
            model, 
            img_tensor.to(device), 
            img_display, 
            misclass['pred_label'], 
            misclass['pred_class'], 
            'integrated_gradients', 
            os.path.join(misclass_dir, "attribution_pred_class.png")
        )
        
        # Generate multi-technique comparison for predicted class
        generate_multi_technique_comparison(
            model,
            img_tensor.to(device),
            img_display,
            misclass['pred_label'],
            misclass['pred_class'],
            os.path.join(misclass_dir, "multi_technique_comparison.png"),
            skip_occlusion=False
        )
    
    # Create a summary CSV
    summary_rows = []
    for i, misclass in enumerate(misclassifications):
        summary_rows.append({
            'id': i+1,
            'true_class': misclass['true_class'],
            'predicted_class': misclass['pred_class'],
            'confidence': misclass['confidence']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "misclassifications_summary.csv"), index=False)
    
    # Create a summary visualization
    plt.figure(figsize=(12, 8))
    confusion_subset = np.zeros((len(class_names), len(class_names)))
    
    for misclass in misclassifications:
        true_idx = misclass['true_label']
        pred_idx = misclass['pred_label']
        confusion_subset[true_idx, pred_idx] += 1
    
    # Plot confusion matrix of misclassifications
    sns.heatmap(confusion_subset, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Misclassification Patterns')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "misclassification_patterns.png"), dpi=300)
    plt.close()

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

def main():
    parser = argparse.ArgumentParser(description='Generate explainable AI visualizations for animal classification model')
    parser.add_argument('--model_path', type=str, default='wildlife_classifier_best.pth', help='Path to trained model')
    parser.add_argument('--class_mapping', type=str, default='class_mapping.json', help='Path to class mapping file')
    parser.add_argument('--data_dir', type=str, default='Set1', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='xai_visualizations', help='Output directory for visualizations')
    parser.add_argument('--image_path', type=str, help='Path to a specific image to analyze (optional)')
    parser.add_argument('--generate_all', action='store_true', help='Generate all visualization types')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for dataset evaluation')
    parser.add_argument('--sample_count', type=int, default=5, help='Number of random samples to visualize per class')
    parser.add_argument('--skip_occlusion', action='store_true', help='Skip the slow Occlusion method')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load class mapping
    class_names, class_to_idx, idx_to_class = load_class_mapping(args.class_mapping)
    
    # Set model type and image size
    model_type = 'efficientnet'
    image_size = 299
    
    print(f"Model type: {model_type}")
    print(f"Image size: {image_size}")
    print(f"Number of classes: {len(class_names)}")
    
    # Load model
    model = load_model(args.model_path, model_type, len(class_names), device)
    
    # If a specific image is provided, analyze just that image
    if args.image_path and os.path.exists(args.image_path):
        print(f"\nAnalyzing image: {args.image_path}")
        img_tensor, img = preprocess_image(args.image_path, image_size)
        
        # Predict class
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor.to(device))
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            _, pred_idx = torch.max(outputs, 1)
            pred_idx = pred_idx.item()
            pred_class = class_names[pred_idx]
            confidence = probs[pred_idx].item()
        
        print(f"Predicted class: {pred_class} with confidence {confidence:.2f}")
        
        # Create output directory for this image
        image_name = os.path.splitext(os.path.basename(args.image_path))[0]
        image_output_dir = os.path.join(args.output_dir, f"single_image_{image_name}")
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Save original image with prediction
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Predicted: {pred_class} ({confidence:.2f})")
        plt.axis('off')
        plt.savefig(os.path.join(image_output_dir, "prediction.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate all visualization types
        print("Generating Integrated Gradients visualization...")
        generate_attribution_visualization(
            model, img_tensor.to(device), img, pred_idx, pred_class, 
            'integrated_gradients', os.path.join(image_output_dir, "integrated_gradients.png")
        )
        
        print("Generating multi-technique comparison...")
        generate_multi_technique_comparison(
            model, img_tensor.to(device), img, pred_idx, pred_class,
            os.path.join(image_output_dir, "multi_technique_comparison.png"),
            skip_occlusion=args.skip_occlusion
        )
        
        print("Generating class activation map...")
        generate_class_activation_maps(
            model, img_tensor.to(device), img, pred_idx, pred_class,
            os.path.join(image_output_dir, "class_activation_map.png")
        )
        
        print("Generating animated attribution (this may take a moment)...")
        generate_animated_attribution(
            model, img_tensor.to(device), img, pred_idx, pred_class,
            os.path.join(image_output_dir, "attribution_animation.gif")
        )
        
        print(f"All visualizations saved to {image_output_dir}")
    
    # Otherwise, analyze the dataset
    else:
        from improved_classification import CameraTrapDataset, prepare_data
        
        print("\nLoading dataset...")
        # Load test data
        data_info = prepare_data(args.data_dir, test_size=0.15, val_size=0.15, validate_images=True)
        
        # Create transform
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create test dataset and dataloader
        test_dataset = CameraTrapDataset(
            data_info['test_paths'],
            data_info['test_labels'],
            transform=test_transform,
            phase='test',
            config=None  # We don't need config here
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        # Generate confusion matrix
        print("Generating confusion matrix...")
        cm, report = generate_confusion_matrix(model, test_loader, device, class_names)
        
        # Save classification report
        with open(os.path.join(args.output_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        # Create per-class visualizations
        if args.generate_all:
            print("\nGenerating per-class visualizations...")
            
            # Group images by class
            class_images = {class_idx: [] for class_idx in range(len(class_names))}
            for i, (img_path, label) in enumerate(zip(data_info['test_paths'], data_info['test_labels'])):
                class_images[label].append(img_path)
            
            # Create visualizations for random samples from each class
            for class_idx, img_paths in class_images.items():
                class_name = class_names[class_idx]
                class_dir = os.path.join(args.output_dir, f"class_{class_idx}_{class_name}")
                os.makedirs(class_dir, exist_ok=True)
                
                # Select random samples
                sample_paths = random.sample(img_paths, min(args.sample_count, len(img_paths)))
                
                for i, img_path in enumerate(sample_paths):
                    # Load and preprocess image
                    img_tensor, img = preprocess_image(img_path, image_size)
                    
                    # Generate attribution visualization
                    generate_attribution_visualization(
                        model, img_tensor.to(device), img, class_idx, class_name,
                        'integrated_gradients', os.path.join(class_dir, f"sample_{i+1}_integrated_gradients.png")
                    )
                    
                    # Generate class activation map
                    generate_class_activation_maps(
                        model, img_tensor.to(device), img, class_idx, class_name,
                        os.path.join(class_dir, f"sample_{i+1}_class_activation_map.png")
                    )
            
            # Create misclassification analysis
            print("\nAnalyzing misclassifications...")
            create_misclassification_analysis(
                model, test_loader, device, class_names,
                output_dir=os.path.join(args.output_dir, 'misclassifications')
            )
        
        print(f"\nAll visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 