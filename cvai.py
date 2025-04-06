from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os
from crewai_tools import SerperDevTool
from crewai import LLM
load_dotenv()

# SETTING UP KEY
os.environ['SERPER_API_KEY'] = "6f2227a931b482b6e1b21298ecc47bb6865beb28"

# LLM
llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0.7,
    api_key="AIzaSyBNOQJ3D5xVYeKt7xokZlQ-zXZrKwGgspE"
)

# TOOLS
# Initialize the tool for internet searching capabilities
serper_tool = SerperDevTool()

# AGENTS
# Define the Animal Species Analysis Agent
animal_species_agent = Agent(
    role="Animal Species Analysis Expert",
    goal=(
        "Analyze the detected animal species to provide comprehensive information about its "
        "characteristics, behavior, habitat preferences, diet, and potential risks. "
        "Include details about whether the animal is endangered, invasive, or native to the region. "
        "Provide information about typical movement patterns, territoriality, and seasonal behaviors "
        "that might explain why the animal is in the detected location."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a wildlife biologist with extensive knowledge of animal species, their behaviors, "
        "and ecological roles. Your expertise helps forest officers understand the animals they encounter "
        "and make informed decisions about how to manage wildlife encounters safely and ethically."
    ),
    tools=[serper_tool],
    llm=llm,
)

# Define the Enhanced Location Analysis Agent
location_analysis_agent = Agent(
    role="Terrain and Location Analysis Expert",
    goal=(
        "Analyze the current detection location in detail, considering terrain features, vegetation, "
        "water sources, and nearby human settlements. Evaluate the current environmental conditions "
        "based on the time and date of detection. "
        "Then, conduct a comprehensive search of nearby areas (within a 5-10 km radius) to identify "
        "potential habitats or territories from which the animal might have originated. "
        "Focus specifically on identifying nearby water bodies, food sources, or natural habitats "
        "that would attract this specific animal species away from its usual territory. "
        "Map potential travel corridors between the detection location and these nearby habitats. "
        "Consider seasonal factors such as drought, mating season, or human activities that might "
        "push animals out of their typical ranges. "
        "Explain why the animal might have traveled from its likely origin point to the detection location "
        "based on resource needs (food, water), territorial behaviors, or environmental pressures."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a geospatial analyst with expertise in wildlife habitat connectivity and movement patterns. "
        "You specialize in analyzing how landscapes influence animal movements across territories. "
        "You have extensive experience mapping wildlife corridors and understanding how environmental "
        "factors and human activities affect animal movement decisions. Your analysis helps predict "
        "not just where animals are coming from, but why they're moving between habitats."
    ),
    tools=[serper_tool],
    llm=llm,
)

# Define the Animal Handling Protocol Agent
animal_handling_agent = Agent(
    role="Wildlife Management and Handling Expert",
    goal=(
        "Develop a safe and ethical handling protocol for the detected animal based on the species information "
        "and location analysis. Provide specific guidance on approach techniques, necessary equipment, "
        "possible sedation requirements, and safe transport methods. Identify potential risks to both "
        "the animal and forest officers. Recommend the best course of action (monitoring, relocation, "
        "medical intervention) based on the animal's condition, location, and species status. "
        "Consider time-of-day factors that might affect handling procedures."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a wildlife veterinarian and animal handling specialist with years of field experience. "
        "You've developed protocols for safely managing wildlife encounters that minimize stress to the "
        "animals while ensuring human safety. Your recommendations balance conservation goals with "
        "practical field realities."
    ),
    tools=[serper_tool],
    llm=llm,
)

# Define the Report Compilation Agent
report_compilation_agent = Agent(
    role="Wildlife Encounter Report Specialist",
    goal=(
        "Compile all information from the species analysis, location assessment, and handling protocol "
        "into a clear, concise, and actionable report for forest officers. Prioritize critical information "
        "that officers need immediately. Organize the report with clear sections, bullet points for key "
        "actions, and visual indicators of risk levels. Include a summary of recommended actions and "
        "any necessary follow-up steps."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in translating complex wildlife management information into practical field "
        "guides. You understand what forest officers need to know and how to present it effectively. "
        "Your reports are known for being comprehensive yet accessible, enabling quick and effective "
        "decision-making in the field."
    ),
    tools=[serper_tool],
    llm=llm,
)

# TASKS
# Define the Animal Species Analysis Task
animal_species_task = Task(
    description=(
        "Research and analyze the detected animal species ({animal_name}) to provide comprehensive "
        "information about its characteristics, behavior patterns, and ecological significance. "
        "Determine if the species is native, invasive, or endangered in the region. "
        "Identify typical movement patterns and behaviors that might explain its presence "
        "at the detection location."
    ),
    expected_output=(
        "A detailed profile of the animal species including:\n"
        "- Species classification and key identifying features\n"
        "- Native range and conservation status\n"
        "- Typical behavior patterns (diurnal/nocturnal, solitary/group, etc.)\n"
        "- Diet and foraging patterns\n"
        "- Breeding and seasonal behaviors\n"
        "- Potential risks to humans or domestic animals\n"
        "- Typical territory size and movement patterns"
    ),
    tools=[serper_tool],
    agent=animal_species_agent,
)

# Define the Enhanced Location Analysis Task
location_analysis_task = Task(
    description=(
        "First, analyze the specific location ({location}) where the animal was detected at {time} in detail. "
        "Describe the current terrain, vegetation, water availability, and proximity to human settlements. "
        "Assess current environmental conditions considering the season, time of day, and recent weather patterns. "
        "Then, identify and analyze all potential nearby habitats within a 5-10 km radius from which the "
        "{animal_name} might have originated, with special focus on natural habitats that match the species' "
        "preferences. Map out likely travel routes between these potential origin points and the detection location, "
        "considering terrain features, barriers, and wildlife corridors. "
        "Explain specific attractants (food, water, shelter) at the detection location that might draw this "
        "animal from its natural habitat. Also consider seasonal pressures or disturbances (drought, fires, "
        "human activity) that might force the animal to seek new territories or resources."
    ),
    expected_output=(
        "A comprehensive location analysis including:\n"
        "1. CURRENT LOCATION ASSESSMENT:\n"
        "   - Detailed description of detection location's terrain and habitat features\n"
        "   - Current environmental conditions (based on season, time, reported weather)\n"
        "   - Resource availability (food, water, shelter) at detection site\n"
        "   - Human activity or disturbances in the area\n\n"
        "2. NEARBY POTENTIAL ORIGIN POINTS:\n"
        "   - Identified natural habitats within 5-10 km that match the species' preferences\n"
        "   - Distance and direction from detection point to each potential origin\n"
        "   - Quality assessment of each habitat as potential home range\n"
        "   - Known wildlife populations or protected areas nearby\n\n"
        "3. MOVEMENT ANALYSIS:\n"
        "   - Likely travel corridors between origin points and detection location\n"
        "   - Natural and artificial barriers that might influence movement patterns\n"
        "   - Specific attractants that might have drawn the animal to the detection location\n"
        "   - Seasonal or environmental factors influencing the animal's movement\n\n"
        "4. RECOMMENDATION:\n"
        "   - Most likely origin point with confidence assessment\n"
        "   - Explanation of why the animal likely traveled to the detection location\n"
        "   - Whether this movement appears to be typical/seasonal or unusual/concerning"
    ),
    tools=[serper_tool],
    agent=location_analysis_agent,
    context=[animal_species_task]
)

# Define the Animal Handling Protocol Task
animal_handling_task = Task(
    description=(
        "Based on the species information and location analysis, develop a comprehensive protocol "
        "for safely handling the {animal_name} detected at {location} at {time}. "
        "Consider the species' specific behaviors, potential risks, conservation status, "
        "and the terrain conditions at the detection site."
    ),
    expected_output=(
        "A detailed handling protocol including:\n"
        "- Recommended approach techniques and distance precautions\n"
        "- Necessary equipment and personnel\n"
        "- Risk assessment for both animal and officers\n"
        "- Containment and transport recommendations if relocation is needed\n"
        "- Special considerations based on time of day/night\n"
        "- Decision tree for different scenarios (animal appears healthy vs. injured, aggressive vs. docile)\n"
        "- Recommended destination for relocation (if applicable)"
    ),
    tools=[serper_tool],
    agent=animal_handling_agent,
    context=[animal_species_task, location_analysis_task]
)

# Define the Report Compilation Task
report_compilation_task = Task(
    description=(
        "Compile the information from species analysis, location assessment, and handling protocol "
        "into a clear, concise, and actionable report for forest officers dealing with the "
        "{animal_name} detected at {location} at {time}."
    ),
    expected_output=(
        "A structured field report with the following sections:\n"
        "1. SUMMARY: Brief overview of situation and key recommendations\n"
        "2. ANIMAL PROFILE: Essential information about the species and its behavior\n"
        "3. LOCATION ASSESSMENT: Analysis of why the animal is there and where it likely came from\n"
        "4. HANDLING PROTOCOL: Step-by-step guidance for approaching and managing the animal, in that also specify what will cause the animal to anger and get be wild and how to approch it calmly.this point is main\n"
        "5. RISK FACTORS: Clearly marked warnings about potential dangers\n"
        "6. NEXT STEPS: Recommendations for follow-up actions\n\n"
        "Report should use simple language, bullet points for key actions, and clear headings."
    ),
    tools=[serper_tool],
    agent=report_compilation_agent,
    context=[animal_species_task, location_analysis_task, animal_handling_task]
)

# CREW
crew = Crew(
    agents=[animal_species_agent, location_analysis_agent, animal_handling_agent, report_compilation_agent],
    tasks=[animal_species_task, location_analysis_task, animal_handling_task, report_compilation_task],
    process=Process.sequential,
)

# MAIN 
def main(animal_name, location, detection_time):
    # Provide input to the Crew and kickoff the process
    result = crew.kickoff(inputs={
        "animal_name": animal_name,
        "location": location,
        "time": detection_time
    })
    print(result)

# FINAL CALLER
if __name__ == "__main__":
    # Example inputs
    animal_name = "Indian Pangolin"
    location = "Outskirts of Sanjay Gandhi National Park, near Borivali East, Mumbai"
    detection_time = "02:30, April 5, 2025"

    
    main(animal_name, location, detection_time)