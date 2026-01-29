import json
import os
import sys

# Add the project root to sys.path so we can import advanced_research
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from advanced_research.main import AdvancedResearch

research_system = AdvancedResearch(
    name="Medical Research Team",
    description="A team of researchers specializing in AI, computer vision, computer graphics",
    max_loops=1,
    output_type="dict",
)

# Run research and get results
result = research_system.run(
    "What are the latest works (within one month) for Programmatic Pattern Design, similar to Garment-Code"
)
print(json.dumps(result, indent=4))

# Save the result to a JSON file
with open("garment_simulation_result.json", "w") as f:
    json.dump(result, f, indent=4)
print("Result saved to garment_simulation_result.json")
