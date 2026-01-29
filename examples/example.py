import json

from advanced_research.main import AdvancedResearch

research_system = AdvancedResearch(
    name="Medical Research Team",
    description="A team of medical researchers who specialize in finding the best treatments for diabetes.",
    max_loops=1,
    output_type="dict",
    director_model_name="claude-3-7-sonnet-20250219",
    worker_model_name="claude-3-7-sonnet-20250219",
)

# Run research and get results
result = research_system.run(
    "What are the latest and highest quality treatments for diabetes? Give me 2 queries"
)
print(json.dumps(result, indent=4))
