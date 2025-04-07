# Wildlife Classification System with Explainable AI and Report Generation

This system combines deep learning-based wildlife classification with explainable AI visualizations and comprehensive reporting. It's designed for wildlife monitoring via camera traps and creates detailed reports for forest officers. [Missouri Camera Traps LILA]([https://lila.science/datasets/missouricameratraps?utm_source=chatgpt.com](https://lila.science/datasets/missouricameratraps#:~:text=This%20data%20set%20contains%20approximately,and%20white-tailed%20deer).))

## Features

- Wildlife species classification using EfficientNet-B3
- Explainable AI visualizations using GradCAM
- Comprehensive HTML reports with model interpretations
- Wildlife safety hazard assessment and handling protocols
- Optional PDF report generation
- Integration with CrewAI for in-depth wildlife analysis
- Various report templates for different species

## Installation

### Prerequisites

1. Python 3.8+ with pip
2. PyTorch and torchvision
3. Required Python packages (see requirements.txt)

### Quick Install

Run the installation script to set up everything automatically:

```bash
python install.py
```

Or manually install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Simple Analysis with HTML Report

To analyze a wildlife image and generate a comprehensive HTML report:

```bash
python analyze_wildlife.py --image_path "path/to/your/image.jpg"
```

To analyze one of the sample images:

```bash
python analyze_wildlife.py --sample [number]
```

### Enhanced Report with Markdown Integration

To generate an enhanced HTML report with integrated markdown analysis:

```bash
python integrate_markdown_report.py --image_path "path/to/your/image.jpg"
```

This combines the standard HTML report with additional AI-generated wildlife analysis in markdown format.

### Safety Protocol & Hazard Assessment

For specialized reports focused on wildlife safety, handling procedures, and risk assessment:

```bash
python wildlife_safety_protocol.py --image_path "path/to/your/image.jpg"
```

This generates a comprehensive safety report that includes:

- Detailed animal hazard assessment
- Safe handling protocols specific to the species
- Risk factors and safety precautions
- Step-by-step guidance for wildlife management personnel

This feature is especially useful for forest officers and wildlife management teams dealing with potentially dangerous wildlife.

### Options for Report Generation

For all report scripts, you can specify:

- `--output_dir` - Directory to save generated reports (default: varies by script)
- `--model_path` - Custom model path (default: wildlife_classifier_best.pth)
- `--class_mapping` - Custom class mapping file (default: class_mapping.json)

PDF Options (for scripts with PDF support):
- `--generate_pdf` - Generate a PDF version of the report

## System Components

1. **Wildlife Classification**
   - `xai_inference_example.py` - Core inference with EfficientNet-B3
   - `explainable_ai.py` - GradCAM visualizations

2. **Reporting System**
   - `inference_to_cvai.py` - Main integration script
   - `integrate_markdown_report.py` - Enhanced reporting with markdown
   - `analyze_wildlife.py` - User-friendly wrapper
   - `wildlife_safety_protocol.py` - Safety-focused reporting

3. **Report Templates**
   - `report_templates/` - Species-specific report templates

## Development

### Adding New Report Templates

To add templates for new species:

1. Create a new file in `report_templates/` named `[species]_report.txt`
2. Follow the template format in existing reports

### Customizing HTML Reports

You can customize the HTML report by modifying the generate_standalone_report function in `inference_to_cvai.py`.

## Troubleshooting

If you encounter issues with report generation:

1. Ensure all dependencies are installed
2. Check file paths and permissions
3. For PDF conversion issues, manually convert HTML to PDF using your web browser
4. For safety protocol features, ensure CrewAI is properly installed

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
