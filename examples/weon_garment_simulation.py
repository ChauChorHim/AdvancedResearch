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

def save_results_to_markdown(results, filename):
    """Saves the research results to a Markdown file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Research Session Results\n\n")
        
        for message in results:
            role = message.get("role", "Unknown")
            content = message.get("content", "")
            
            # Format based on role
            if role == "human":
                f.write(f"## 🧑 User Request\n\n{content}\n\n")
            elif role == "Director-Agent":
                f.write(f"## 🤖 Research Report\n\n{content}\n\n")
            else:
                f.write(f"## {role}\n\n{content}\n\n")
            
            f.write("---\n\n")
    
    print(f"Results saved to {filename}")

# Save to Markdown
save_results_to_markdown(result, "garment_simulation_result.md")
