#!/usr/bin/env python3
"""
Wildlife Safety Protocol Generator

This script generates comprehensive safety protocols and risk assessments for wildlife encounters.
It identifies animal species from camera trap images and uses the CrewAI system to generate
detailed reports on potential hazards and appropriate handling protocols.

Usage:
    python wildlife_safety_protocol.py --image_path "path/to/image.jpg" --output_dir "output_directory"
"""

import os
import sys
import json
import argparse
import datetime
import webbrowser
import subprocess
from pathlib import Path

# Import necessary modules
sys.path.append(os.getcwd())

# Try importing CrewAI module
CREWAI_AVAILABLE = False
try:
    from cvai import main as cvai_main
    CREWAI_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"CrewAI integration not available: {e}")
    print("CrewAI is required for full functionality. Please install the required packages.")
    print("Required packages: crewai, crewai_tools")
    print("To install: pip install crewai crewai_tools")

def run_wildlife_analysis(image_path, output_dir, model_path="wildlife_classifier_best.pth", class_mapping="class_mapping.json"):
    """
    Run the wildlife image analysis pipeline
    
    Args:
        image_path: Path to the wildlife image
        output_dir: Directory to save output files
        model_path: Path to the classification model
        class_mapping: Path to the class mapping file
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    try:
        # Build the command to run inference_to_cvai.py
        cmd = [
            "python", "inference_to_cvai.py",
            "--image_path", image_path,
            "--output_dir", output_dir,
            "--model_path", model_path,
            "--class_mapping", class_mapping
        ]
        
        # Run the command
        print(f"Running wildlife image analysis on: {image_path}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Load inference results from JSON
        results_file = os.path.join(output_dir, "inference_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Error: Results file not found at {results_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running wildlife analysis: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def generate_safety_hazard_assessment(animal_name, location, detection_time):
    """
    Generate a comprehensive safety hazard assessment using CrewAI
    
    Args:
        animal_name: Name of the detected animal
        location: Location where the animal was detected
        detection_time: Time when the animal was detected
        
    Returns:
        String with the safety assessment report
    """
    if not CREWAI_AVAILABLE:
        return None
        
    print(f"\n{'='*80}")
    print(f"GENERATING WILDLIFE SAFETY & HANDLING PROTOCOL")
    print(f"{'='*80}")
    print(f"Animal: {animal_name}")
    print(f"Location: {location}")
    print(f"Detection Time: {detection_time}")
    print(f"{'='*80}\n")
    
    # Call CrewAI with specific focus on safety hazards and handling
    report = cvai_main(animal_name, location, detection_time)
    return report

def extract_safety_sections(full_report):
    """
    Extract safety and handling protocol sections from the full report
    
    Args:
        full_report: Complete report from CrewAI
        
    Returns:
        Dictionary with safety sections
    """
    if not full_report:
        return None
        
    safety_info = {}
    
    # Define section markers
    sections = [
        ("RISK FACTORS", "RISK FACTORS", "NEXT STEPS"),
        ("HANDLING PROTOCOL", "HANDLING PROTOCOL", "RISK FACTORS"),
        ("ANIMAL PROFILE", "ANIMAL PROFILE", "LOCATION ASSESSMENT"),
        ("SUMMARY", "SUMMARY", "ANIMAL PROFILE")
    ]
    
    # Extract each section
    for section_name, start_marker, end_marker in sections:
        start_idx = full_report.find(start_marker)
        if start_idx == -1:
            continue
            
        # Find the end of the section
        end_idx = full_report.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            # If end marker not found, take until the end
            section_content = full_report[start_idx:]
        else:
            section_content = full_report[start_idx:end_idx]
            
        safety_info[section_name] = section_content.strip()
    
    return safety_info

def generate_enhanced_html_report(inference_results, safety_info, output_dir):
    """
    Generate an enhanced HTML report with safety and handling protocols
    
    Args:
        inference_results: Results from wildlife image analysis
        safety_info: Safety and handling information
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated HTML report
    """
    animal_name = inference_results["prediction"]["clean_name"]
    confidence = inference_results["prediction"]["confidence"] * 100
    location = inference_results["metadata"]["location"]
    detection_time = inference_results["metadata"]["detection_time"]
    
    # Path to output HTML file
    html_path = os.path.join(output_dir, "wildlife_safety_report.html")
    
    # Create HTML content with safety focus
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wildlife Safety & Handling Protocol: {animal_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #b71c1c; color: white; padding: 20px; text-align: center; }}
            .content {{ margin-top: 20px; }}
            .prediction {{ background-color: #f2f2f2; padding: 15px; margin-bottom: 20px; }}
            .safety-alert {{ background-color: #ffebee; border-left: 4px solid #b71c1c; padding: 15px; margin: 20px 0; }}
            .handling-protocol {{ background-color: #e8f5e9; border-left: 4px solid #388e3c; padding: 15px; margin: 20px 0; }}
            .animal-profile {{ background-color: #e3f2fd; border-left: 4px solid #1976d2; padding: 15px; margin: 20px 0; }}
            .visualization {{ display: flex; justify-content: space-between; margin-top: 20px; }}
            .viz-item {{ width: 48%; text-align: center; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; }}
            th {{ background-color: #616161; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .footer {{ margin-top: 30px; font-size: 0.8em; color: #777; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .danger {{ color: #b71c1c; font-weight: bold; }}
            .warning {{ color: #f57f17; font-weight: bold; }}
            .safe {{ color: #388e3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Wildlife Safety & Handling Protocol</h1>
            <h2>{animal_name} - {confidence:.2f}% Confidence</h2>
        </div>
        
        <div class="content">
            <div class="prediction">
                <h2>Detection Information</h2>
                <p><strong>Species:</strong> {animal_name}</p>
                <p><strong>Location:</strong> {location}</p>
                <p><strong>Detection Time:</strong> {detection_time}</p>
                <p><strong>AI Confidence:</strong> {confidence:.2f}%</p>
            </div>
            
            <div class="visualization">
                <div class="viz-item">
                    <h3>Original Detection Image</h3>
                    <img src="prediction.png" alt="Wildlife Detection">
                </div>
                <div class="viz-item">
                    <h3>AI Feature Identification (GradCAM)</h3>
                    <img src="gradcam.png" alt="GradCAM Visualization">
                </div>
            </div>
    """
    
    # Add safety information if available
    if safety_info:
        if "SUMMARY" in safety_info:
            html_content += f"""
            <h2>SUMMARY</h2>
            <div class="safety-alert">
                {safety_info["SUMMARY"]}
            </div>
            """
            
        if "ANIMAL PROFILE" in safety_info:
            html_content += f"""
            <h2>ANIMAL PROFILE</h2>
            <div class="animal-profile">
                {safety_info["ANIMAL PROFILE"]}
            </div>
            """
            
        if "HANDLING PROTOCOL" in safety_info:
            html_content += f"""
            <h2>HANDLING PROTOCOL</h2>
            <div class="handling-protocol">
                {safety_info["HANDLING PROTOCOL"]}
            </div>
            """
            
        if "RISK FACTORS" in safety_info:
            html_content += f"""
            <h2>RISK FACTORS & SAFETY HAZARDS</h2>
            <div class="safety-alert">
                {safety_info["RISK FACTORS"]}
            </div>
            """
    else:
        # If no CrewAI safety info available, provide comprehensive safety guidelines specific to the animal
        # Specialized content for Wild Boar
        if "Wild Boar" in animal_name or "wild boar" in animal_name.lower() or "boar" in animal_name.lower():
            html_content += f"""
            <h2>SUMMARY</h2>
            <div class="safety-alert">
                <p>A wild boar (<i>Sus scrofa</i>) was detected at camera trap location {location} on {detection_time}. This report outlines the species profile, location assessment, handling protocol, risk factors, and next steps for forest officers. Prioritize a cautious approach and consult a wildlife veterinarian before attempting capture. Relocation to a suitable habitat is the preferred option if the animal is healthy and non-aggressive.</p>
            </div>
            
            <h2>ANIMAL PROFILE</h2>
            <div class="animal-profile">
                <ul>
                    <li><strong>Species:</strong> Wild Boar (<i>Sus scrofa</i>)</li>
                    <li><strong>Identifying Features:</strong> Robust build, coarse dark brown to black bristly hair, long snout, powerful tusks (longer in males), shoulder hump.</li>
                    <li><strong>Behavior:</strong> Primarily crepuscular (active at dawn and dusk), but may be more nocturnal near human activity. Lives in sounders (groups of females and young), adult males are generally solitary except during breeding season. Omnivorous diet (roots, tubers, fruits, insects, etc.). Can be aggressive, especially when threatened or protecting young.</li>
                    <li><strong>Potential Risks:</strong> Serious injury from tusks, disease transmission.</li>
                </ul>
            </div>
            
            <h2>LOCATION ASSESSMENT</h2>
            <div class="animal-profile">
                <p><em>(Assumptions made due to lack of specific data on the detection location)</em></p>
                <ul>
                    <li><strong>Habitat:</strong> Assumed mixed deciduous forest with open grassland near a water source. Gently rolling terrain.</li>
                    <li><strong>Resource Availability:</strong> Abundant food (roots, tubers, insects) and water.</li>
                    <li><strong>Human Disturbance:</strong> Moderate level assumed, potentially influencing boar's behavior.</li>
                    <li><strong>Likely Origin:</strong> A deciduous forest approximately 3 km Northwest of the detection site is the most likely origin point based on typical boar habitat preference. This is a high-confidence assumption, requiring verification.</li>
                </ul>
            </div>
            
            <h2>HANDLING PROTOCOL</h2>
            <div class="handling-protocol">
                <p class="danger"><strong>WARNING:</strong> Wild boars are potentially dangerous. Prioritize officer safety. Improper handling can lead to serious injury. This protocol requires experienced personnel and veterinary consultation.</p>
                
                <h3>Team & Equipment</h3>
                <ul>
                    <li><strong>Team:</strong> Minimum 4 experienced personnel: Lead veterinarian/handler, 2 assistants, 1 driver/logistician.</li>
                    <li><strong>Equipment:</strong> Dart gun with appropriate darts, immobilization drugs (tiletamine-zolazepam, ketamine, xylazine - dosages determined by veterinarian), syringes, needles, protective gear (gloves, long sleeves, eye protection, boots, ideally boar-resistant vests), transport crate, measuring tape, scales (if possible), telemetry device (if available), radio collars (if relocation is required), first-aid kit.</li>
                </ul>
                
                <h3>Approach & Handling</h3>
                <ul>
                    <li><strong>Approach:</strong> Slow, deliberate approach from a safe distance (minimum 20 meters). Use natural cover. Observe boar's behavior. Avoid sudden movements. Approach during crepuscular hours if possible.</li>
                    <li><strong>Immobilization:</strong> Dart gun from a safe distance if boar is calm. Other methods (net gun) should only be used by highly experienced personnel. Continuously monitor vital signs.</li>
                    <li><strong>Handling & Transport:</strong> Secure limbs, minimize stress, monitor vital signs during transport.</li>
                    <li><strong>Post-Capture:</strong> Thorough physical exam, collect samples if needed. Treatment as necessary.</li>
                </ul>
                
                <h3>Decision Tree</h3>
                <ul>
                    <li><strong>Healthy, Non-Aggressive:</strong> Relocation to a suitable habitat.</li>
                    <li><strong>Healthy, Aggressive:</strong> Relocation to a remote area.</li>
                    <li><strong>Injured/Ill:</strong> Veterinary treatment. Euthanasia may be a last resort (by veterinarian only).</li>
                </ul>
                
                <h3>Relocation & Documentation</h3>
                <ul>
                    <li><strong>Relocation:</strong> Select a site with abundant food, cover, minimal human activity, and existing boar populations (if possible). Coordinate with relevant authorities. Monitor post-release.</li>
                    <li><strong>Documentation:</strong> Thoroughly document all aspects of the procedure.</li>
                </ul>
            </div>
            
            <h2>RISK FACTORS & SAFETY HAZARDS</h2>
            <div class="safety-alert">
                <ul>
                    <li class="danger"><strong>HIGH:</strong> Injury to officers from boar's tusks and aggression.</li>
                    <li class="warning"><strong>MEDIUM:</strong> Stress to the boar, risk of disease transmission.</li>
                    <li class="safe"><strong>LOW:</strong> Relocation failure (if attempted).</li>
                </ul>
            </div>
            
            <h2>NEXT STEPS</h2>
            <div class="safety-alert">
                <ul>
                    <li><strong>Immediate:</strong> Assess the situation remotely to determine the boar's behavior and proximity to human activity. Prepare equipment and personnel for potential capture and relocation. Contact a wildlife veterinarian for consultation on immobilization drugs and protocols.</li>
                    <li><strong>Short-term (within 24-48 hours):</strong> Attempt capture and relocation if deemed safe and necessary.</li>
                    <li><strong>Long-term:</strong> Monitor the area for further boar sightings. Evaluate the effectiveness of the relocation (if performed) through monitoring (telemetry if available). Consider longer-term population management strategies if the boar population is problematic.</li>
                </ul>
            </div>
            
            <p class="danger"><strong>DISCLAIMER:</strong> This report is based on general knowledge and assumptions due to limited information about the specific location. On-site assessment is crucial. Always prioritize safety and consult with experts.</p>
            """
        # Specialized content for Ocelot
        elif "Ocelot" in animal_name or "ocelot" in animal_name.lower():
            html_content += f"""
            <h2>SUMMARY</h2>
            <div class="safety-alert">
                <p>An ocelot (<i>Leopardus pardalis</i>) was detected at camera trap location {location} on {detection_time}. This report provides species information, handling protocols, and safety considerations. Ocelots are protected under endangered species legislation in many areas. Any intervention must prioritize both human safety and animal welfare. Non-invasive monitoring is strongly recommended over direct handling or relocation unless absolutely necessary.</p>
            </div>
            
            <h2>ANIMAL PROFILE</h2>
            <div class="animal-profile">
                <ul>
                    <li><strong>Species:</strong> Ocelot (<i>Leopardus pardalis</i>)</li>
                    <li><strong>Conservation Status:</strong> Listed as "Least Concern" by IUCN, but nationally protected in many countries. U.S. populations are federally endangered.</li>
                    <li><strong>Identifying Features:</strong> Medium-sized wild cat (15-35 lbs), distinctive spotted and striped coat pattern, solid black markings on a gold/tawny background, white underside, slightly rounded ears with white spot.</li>
                    <li><strong>Behavior:</strong> Primarily nocturnal and solitary. Excellent climbers and swimmers. Primarily terrestrial hunters. Highly territorial with home ranges of 1-5 square miles for females and up to 25 square miles for males.</li>
                    <li><strong>Potential Risks:</strong> Can be defensive if cornered or threatened. Capable of causing significant injury through teeth and claws, though rarely aggressive toward humans without provocation.</li>
                </ul>
            </div>
            
            <h2>LOCATION ASSESSMENT</h2>
            <div class="animal-profile">
                <p><em>(Assumptions made due to lack of specific data on the detection location)</em></p>
                <ul>
                    <li><strong>Habitat:</strong> Assumed to be dense forest or brushland with good cover, likely near water source. Ocelots prefer areas with thick vegetation for hunting and shelter.</li>
                    <li><strong>Resource Availability:</strong> Presumed adequate prey base (small to medium mammals, birds, reptiles) and water sources.</li>
                    <li><strong>Human Disturbance:</strong> Likely minimal, as ocelots typically avoid areas with high human activity.</li>
                    <li><strong>Territory Considerations:</strong> This is likely part of the animal's established territory; ocelots are highly territorial and tend to maintain stable home ranges.</li>
                </ul>
            </div>
            
            <h2>HANDLING PROTOCOL</h2>
            <div class="handling-protocol">
                <p class="danger"><strong>WARNING:</strong> Ocelots are protected wildlife and should only be handled by professionals with proper permits and expertise. Direct handling should be limited to emergency situations (injury, public safety risk) and authorized wildlife rehabilitation.</p>
                
                <h3>Team & Equipment</h3>
                <ul>
                    <li><strong>Team:</strong> Minimum 3 specialized personnel: Wildlife veterinarian with felid experience, experienced wildlife biologist, and trained handler/assistant.</li>
                    <li><strong>Equipment:</strong> 
                        <ul>
                            <li>Chemical immobilization equipment (dart gun/blowpipe, appropriate sedatives - typically ketamine/medetomidine combination)</li>
                            <li>Transport kennel (airline-approved, solid sides)</li>
                            <li>Kevlar or leather handling gloves</li>
                            <li>Catch poles, Y-poles, nets</li>
                            <li>Monitoring equipment (thermometer, pulse oximeter, stethoscope)</li>
                            <li>Eye protection and face shields</li>
                            <li>First aid kit (for humans and animal)</li>
                            <li>Sampling kit if appropriate (microchips, DNA samples, etc.)</li>
                        </ul>
                    </li>
                </ul>
                
                <h3>Approach & Handling Techniques</h3>
                <ul>
                    <li><strong>Initial Assessment:</strong> Observe from a safe distance (minimum 30 meters) using binoculars. Note behavior, any signs of injury or illness, and proximity to human settlements.</li>
                    <li><strong>Approach Strategy:</strong> Move slowly and quietly. Avoid direct eye contact which can be perceived as threatening. Approach from downwind if possible.</li>
                    <li><strong>Immobilization:</strong> Chemical immobilization is the preferred method for any direct handling. Dosage must be calculated by a wildlife veterinarian based on estimated weight.</li>
                    <li><strong>Physical Handling:</strong> Once sedated, minimize handling time. Cover eyes with a cloth to reduce stress. Monitor vital signs continuously. Maintain normal body temperature.</li>
                    <li><strong>Recovery:</strong> Allow recovery in a quiet, dark transport container. Do not release until fully recovered from sedation.</li>
                </ul>
                
                <h3>Decision Tree</h3>
                <ul>
                    <li><strong>Healthy, In Appropriate Habitat:</strong> Implement non-invasive monitoring. Do not disturb or relocate.</li>
                    <li><strong>Healthy, In Inappropriate Location:</strong> Chemical immobilization and relocation to nearest appropriate habitat within its presumed home range.</li>
                    <li><strong>Injured/Sick:</strong> Chemical immobilization, veterinary assessment, and treatment. Transfer to wildlife rehabilitation facility if necessary.</li>
                    <li><strong>Direct Threat to Human Safety:</strong> Contact appropriate wildlife authorities immediately. Secure perimeter and keep people away. Chemical immobilization and relocation.</li>
                </ul>
                
                <h3>Agitation Triggers & Calming Techniques</h3>
                <ul>
                    <li><strong>Triggers for Defensive Behavior:</strong>
                        <ul>
                            <li>Cornering or blocking escape routes</li>
                            <li>Direct eye contact or frontal approach</li>
                            <li>Loud noises or sudden movements</li>
                            <li>Presence of dogs or other perceived threats</li>
                            <li>Proximity to den sites or young (if present)</li>
                        </ul>
                    </li>
                    <li><strong>Calming Techniques:</strong>
                        <ul>
                            <li>Maintain distance and provide clear escape routes</li>
                            <li>Move slowly and speak in low, quiet tones</li>
                            <li>Avoid direct eye contact</li>
                            <li>Use visual barriers during handling</li>
                            <li>Minimize handling time and personnel present</li>
                        </ul>
                    </li>
                </ul>
            </div>
            
            <h2>RISK FACTORS & SAFETY HAZARDS</h2>
            <div class="safety-alert">
                <ul>
                    <li class="danger"><strong>HIGH:</strong> 
                        <ul>
                            <li>Injuries from claws and teeth during immobilization attempts or handling</li>
                            <li>Stress-induced physiological complications during capture (hyperthermia, capture myopathy)</li>
                        </ul>
                    </li>
                    <li class="warning"><strong>MEDIUM:</strong> 
                        <ul>
                            <li>Zoonotic disease transmission (rare but possible)</li>
                            <li>Legal consequences of improper handling of protected species</li>
                            <li>Adverse reaction to chemical immobilization agents</li>
                        </ul>
                    </li>
                    <li class="safe"><strong>LOW:</strong> 
                        <ul>
                            <li>Unprovoked attack on humans (extremely rare)</li>
                            <li>Failed relocation attempt</li>
                        </ul>
                    </li>
                </ul>
            </div>
            
            <h2>NEXT STEPS</h2>
            <div class="safety-alert">
                <ul>
                    <li><strong>Immediate (within 24 hours):</strong>
                        <ul>
                            <li>Verify presence through additional camera trap monitoring if possible</li>
                            <li>Notify relevant wildlife authorities of the sighting</li>
                            <li>Secure appropriate permits if intervention may be necessary</li>
                            <li>Restrict human access to the area</li>
                        </ul>
                    </li>
                    <li><strong>Short-term (within 1 week):</strong>
                        <ul>
                            <li>Establish non-invasive monitoring protocol (additional camera traps)</li>
                            <li>Assess habitat quality and connectivity</li>
                            <li>Educate local communities about the presence of ocelots and appropriate behavior</li>
                        </ul>
                    </li>
                    <li><strong>Long-term:</strong>
                        <ul>
                            <li>Contribute data to regional conservation efforts</li>
                            <li>Consider habitat protection or enhancement measures</li>
                            <li>Monitor for evidence of breeding activity</li>
                        </ul>
                    </li>
                </ul>
            </div>
            
            <p class="danger"><strong>DISCLAIMER:</strong> This report is based on general knowledge of ocelot biology and behavior. Site-specific assessment is essential. Always prioritize both human safety and animal welfare. Handling of endangered species may require special permits - verify legal requirements before any intervention.</p>
            """
        else:
            # General fallback for other animals
            html_content += f"""
            <h2>SAFETY PRECAUTIONS - GENERAL GUIDANCE</h2>
            <div class="safety-alert">
                <p class="danger">⚠️ IMPORTANT: This is general guidance only. Without CrewAI integration, specific safety protocols for {animal_name} cannot be provided.</p>
                
                <h3>General Wildlife Safety Guidelines:</h3>
                <ul>
                    <li>Never approach or attempt to handle wild animals without proper training and equipment.</li>
                    <li>Maintain a safe distance from all wildlife.</li>
                    <li>Do not feed wild animals.</li>
                    <li>Be aware of your surroundings when in wildlife habitats.</li>
                    <li>Contact local wildlife authorities for proper handling procedures.</li>
                </ul>
                
                <p>For detailed safety protocols and hazard assessments, please install the CrewAI integration.</p>
            </div>
            """
    
    # Complete the HTML content
    html_content += f"""
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Wildlife Safety Protocol System - v1.0</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Enhanced safety report generated: {html_path}")
    return html_path

def main():
    """Main function to run the wildlife safety protocol generator"""
    parser = argparse.ArgumentParser(description="Generate wildlife safety protocols and hazard assessments")
    parser.add_argument("--image_path", required=True, help="Path to the wildlife image")
    parser.add_argument("--output_dir", default="wildlife_safety_report", help="Directory to save output files")
    parser.add_argument("--model_path", default="wildlife_classifier_best.pth", help="Path to the model file")
    parser.add_argument("--class_mapping", default="class_mapping.json", help="Path to the class mapping file")
    parser.add_argument("--no_browser", action="store_true", help="Don't open report in browser")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Run wildlife analysis
    inference_results = run_wildlife_analysis(
        args.image_path,
        args.output_dir,
        args.model_path,
        args.class_mapping
    )
    
    if not inference_results:
        print("Wildlife analysis failed. Exiting.")
        return
    
    # Extract animal information
    animal_name = inference_results["prediction"]["clean_name"]
    location = inference_results["metadata"]["location"]
    detection_time = inference_results["metadata"]["detection_time"]
    
    # Step 2: Generate safety and handling protocol
    safety_report = None
    if CREWAI_AVAILABLE:
        safety_report = generate_safety_hazard_assessment(
            animal_name,
            location,
            detection_time
        )
        
        # Extract safety sections
        safety_info = extract_safety_sections(safety_report)
    else:
        safety_info = None
        print("CrewAI not available - only basic safety guidelines will be provided")
    
    # Step 3: Generate enhanced HTML report
    html_path = generate_enhanced_html_report(
        inference_results,
        safety_info,
        args.output_dir
    )
    
    # Open in browser if requested
    if html_path and not args.no_browser:
        print(f"Opening safety report in browser: {html_path}")
        webbrowser.open('file://' + os.path.abspath(html_path))

if __name__ == "__main__":
    main() 