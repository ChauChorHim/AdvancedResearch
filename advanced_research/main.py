import concurrent.futures
import os
import uuid
from datetime import datetime
from typing import Any, List, Optional

import orjson
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from swarms.prompts.agent_conversation_aggregator import (
    AGGREGATOR_SYSTEM_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    HistoryOutputType,
    history_output_formatter,
)
from swarms.utils.litellm_tokenizer import count_tokens
from swarms_tools import exa_search

from advanced_research.prompts import (
    get_citation_prompt,
    get_evaluation_prompt,
    get_orchestrator_prompt,
    get_subagent_prompt,
)
from advanced_research.search_tools import (
    duckduckgo_search_tool,
    gemini_search_tool,
)

load_dotenv()


model_name = os.getenv("WORKER_MODEL_NAME", "gemini/gemini-2.0-flash")
max_tokens = int(os.getenv("WORKER_MAX_TOKENS", 8000))
exa_search_num_results = int(os.getenv("EXA_SEARCH_NUM_RESULTS", 2))
exa_search_max_characters = int(
    os.getenv("EXA_SEARCH_MAX_CHARACTERS", 100)
)
director_model_name = os.getenv(
    "DIRECTOR_MODEL_NAME", "gemini/gemini-2.0-pro-exp-02-05"
)


class TokenTracker:
    """
    Singleton-ish class to track token usage across the research session.
    """

    _instance = None
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def track(
        cls,
        input_str: str,
        output_str: str,
        agent_name: str = "Agent",
    ):
        try:
            in_count = count_tokens(input_str)
            out_count = count_tokens(output_str)
            cls.input_tokens += in_count
            cls.output_tokens += out_count
            cls.total_tokens += in_count + out_count
            logger.info(
                f"[{agent_name}] Tokens - Input: {in_count}, Output: {out_count}, Total: {in_count + out_count}"
            )
        except Exception as e:
            logger.warning(f"Failed to track tokens: {e}")

    @classmethod
    def get_stats(cls):
        return {
            "input_tokens": cls.input_tokens,
            "output_tokens": cls.output_tokens,
            "total_tokens": cls.total_tokens,
        }

    @classmethod
    def reset(cls):
        cls.input_tokens = 0
        cls.output_tokens = 0
        cls.total_tokens = 0


def generate_id():
    return f"AdvancedResearch-{uuid.uuid4()}-time-{datetime.now().strftime('%Y%m%d%H%M%S')}"


# Schema
class AdvancedResearchAdditionalConfig(BaseModel):
    worker_model_name: str = Field(
        default=model_name,
        description="The model name to use for the worker agent.",
    )
    worker_max_tokens: int = Field(
        default=max_tokens,
        description="The maximum number of tokens to use for the worker agent.",
    )
    exa_search_num_results: int = Field(
        default=exa_search_num_results,
        description="The number of results to return from the Exa search tool.",
    )
    exa_search_max_characters: int = Field(
        default=exa_search_max_characters,
        description="The maximum number of characters to return from the Exa search tool.",
    )
    worker_max_loops: int = Field(
        default=3,
        description="The maximum number of loops to use for the worker agent.",
    )
    enable_evaluation: bool = Field(
        default=True,
        description="Whether to enable the evaluation agent.",
    )
    enable_citation: bool = Field(
        default=True,
        description="Whether to enable the citation agent.",
    )


schema = AdvancedResearchAdditionalConfig()


def summarization_agent(
    model_name: str | None = model_name,
    task: str | None = None,
    img: str = None,
    **kwargs: Any,
) -> str:
    """
    Summarization agent for generating a concise summary of research findings.

    Args:
        model_name (str, optional): The name of the language model to use for summarization.
            Defaults to "claude-3-7-sonnet-20250219".
        task (str, optional): The research findings or content to be summarized.
            Should be a string describing the summarization task or the text to summarize.
        max_tokens (int, optional): The maximum number of tokens to generate in the summary.
            Defaults to 1000.
        img (str, optional): Optional image input for multimodal summarization, if supported.
        **kwargs: Additional keyword arguments passed to the Agent.

    Returns:
        str: The generated summary of the research findings.
    """
    agent = Agent(
        agent_name="Report-Generator-Agent",
        agent_description="An agent that generates a concise summary of research findings.",
        system_prompt=AGGREGATOR_SYSTEM_PROMPT,
        model_name=model_name,
        max_loops=1,
    )
    return agent.run(task=task, img=img)


def create_json_file(data: dict, file_name: str):
    """
    Creates or updates a JSON file with the provided data.

    If the file already exists and contains a JSON object, the function updates it with the new data.
    If the file does not exist or contains invalid data, it creates a new file with the provided data.

    Args:
        data (dict): The data to write to the JSON file.
        file_name (str): The name (path) of the JSON file to create or update.

    Returns:
        None
    """
    # Check if file exists and load existing data
    if os.path.exists(file_name):
        try:
            with open(file_name, "rb") as f:
                existing_data = orjson.loads(f.read())
        except Exception:
            existing_data = {}
        if isinstance(existing_data, dict):
            existing_data.update(data)
            data_to_write = existing_data
        else:
            data_to_write = data
    else:
        data_to_write = data

    with open(file_name, "wb") as f:
        f.write(
            orjson.dumps(data_to_write, option=orjson.OPT_INDENT_2)
        )


def evaluate_research(query: str, research_output: str) -> str:
    """
    Evaluates the research output using an LLM-as-judge approach.

    Args:
        query (str): The original research query.
        research_output (str): The output generated by the worker agent.

    Returns:
        str: The evaluation result.
    """
    evaluation_task = f"""
    Original Query: {query}
    
    Research Output to Evaluate:
    {research_output}
    
    Please evaluate the research output based on the provided prompt.
    """

    evaluator = Agent(
        agent_name="Evaluator-Agent",
        system_prompt=get_evaluation_prompt(),
        model_name=schema.worker_model_name,
        max_loops=1,
        max_tokens=2000,
    )
    result = evaluator.run(task=evaluation_task)
    TokenTracker.track(
        evaluation_task + evaluator.system_prompt,
        result,
        "Evaluator-Agent",
    )
    return result


def run_agent(i: int, query: str):
    """
    Runs a worker search agent to process a research query.
    """
    search_provider = os.getenv("SEARCH_PROVIDER", "exa").lower()
    if search_provider == "gemini":
        search_tool = gemini_search_tool
    elif search_provider in ["ddg", "duckduckgo"]:
        search_tool = duckduckgo_search_tool
    else:
        search_tool = exa_search
    """
    Runs a worker search agent to process a research query.

    This function instantiates a Swarms Agent with a synthesis prompt and the Exa search tool,
    then executes the agent on the provided query.

    Args:
        i (int): The index or identifier for the agent instance (used in the agent's name).
        query (str): The research query to be processed by the agent.

    Returns:
        str: The output generated by the agent for the given query.
    """
    # Can also put agent judge here
    agent = Agent(
        agent_name=f"Worker-Search-Agent-{i}",
        system_prompt=get_subagent_prompt(
            strategy_context="general",
            max_loops=schema.worker_max_loops,
        ),
        model_name=schema.worker_model_name,
        max_loops=schema.worker_max_loops,
        max_tokens=schema.worker_max_tokens,
        tools=[search_tool],
        tool_call_summary=True,
    )
    result = agent.run(task=query)
    TokenTracker.track(
        query + agent.system_prompt,
        result,
        f"Worker-Search-Agent-{i}",
    )

    if schema.enable_evaluation:
        evaluation = evaluate_research(query, result)
        return f"{result}\n\n--- Evaluation ---\n{evaluation}"

    return result


def execute_worker_search_agents(
    queries: list[str],
) -> str:
    """
    Executes multiple worker search agents in sequence, each responsible for handling a single research query.

    This function is designed to automate the process of running multiple independent search agents (one per query)
    using the Swarms Agent framework. Each agent is initialized with a custom system prompt tailored to its specific
    query, and is equipped with the Exa search tool for web research. The agents are run in parallel using a thread pool,
    and their outputs are collected and returned as a list.

    Args:
        queries (list[str]):
            A list of research queries (strings) to be processed. Each query will be handled by a separate agent.

    Returns:
        str:
            A string containing the output from all worker search agents, concatenated together.

    Workflow:
        1. For each query in the input list:
            a. Instantiate a Swarms Agent with:
                - A unique agent name based on the query.
                - A system prompt generated by `get_subagent_prompt`, customized for the query, number of results, and max characters.
                - The specified model ("claude-3-7-sonnet-20250219").
                - A single loop (max_loops=1) to ensure one-shot execution.
                - A generous token limit (max_tokens=8000) to accommodate detailed outputs.
                - The Exa search tool enabled for web research.
            b. Run the agent with the query as its task.
            c. Collect the agent's output (typically a summary or structured research findings).
        2. Return a list of all agent outputs.

    Notes:
        - This function runs agents in parallel using ThreadPoolExecutor to maximize throughput.
        - The function assumes that `get_subagent_prompt` and `exa_search` are properly defined and imported.
        - The agent's output format depends on the system prompt and the agent's implementation.
        - Useful for orchestrating multi-query research tasks in advanced research pipelines.

    Example:
        >>> queries = ["What are the latest advances in quantum computing?", "Summarize recent AI safety research."]
        >>> results = execute_worker_search_agents(queries)
        >>> print(results[0])  # Output from the first query's agent
        >>> print(results[1])  # Output from the second query's agent
    """

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count()
    ) as executor:
        futures = [
            executor.submit(run_agent, i, query)
            for i, query in enumerate(queries)
        ]
        results = [
            future.result()
            for future in concurrent.futures.as_completed(futures)
        ]

    return " ".join(results)


def generate_citations(report: str) -> str:
    """
    Generates citations for the research report using a specialized Citation Agent.

    Args:
        report (str): The research report to be cited.

    Returns:
        str: The report with proper citations and references.
    """
    citation_agent = Agent(
        agent_name="Citation-Agent",
        system_prompt=get_citation_prompt(),
        model_name=schema.worker_model_name,
        max_loops=1,
        max_tokens=schema.worker_max_tokens,
    )
    result = citation_agent.run(task=report)
    TokenTracker.track(
        report + citation_agent.system_prompt,
        result,
        "Citation-Agent",
    )
    return result


class ResearchMemory(BaseModel):
    """
    ResearchMemory stores the current state of the research process,
    including the plan and key findings.
    """

    plan: str = Field(
        default="", description="The current research plan."
    )
    findings: List[str] = Field(
        default_factory=list, description="List of key findings."
    )
    current_phase: str = Field(
        default="init", description="Current phase of research."
    )


def create_director_agent(
    agent_name: str = "Director-Agent",
    model_name: str = director_model_name,
    task: str | None = None,
    img: Optional[str] = None,
    max_loops: int = 1,
    extra_tools: List[Any] = None,
    *args,
    **kwargs,
):
    """
    Create a director agent for the advanced research system.

    Args:
        agent_name (str): Name of the director agent. Default is "Director-Agent".
        model_name (str): Model to use for the agent. Default is "claude-3-7-sonnet-20250219".
        task (str | None): The research task or instruction for the agent to execute.
        max_tokens (int): Maximum number of tokens for the agent's output. Default is 8000.
        img (Optional[str]): Optional image input for the agent.
        extra_tools (List[Any]): Additional tools to provide to the agent.
        **kwargs: Additional keyword arguments.

    Returns:
        str: The output from the director agent after running the specified task.
    """
    tools = [execute_worker_search_agents]
    if extra_tools:
        tools.extend(extra_tools)

    director_agent = Agent(
        agent_name=agent_name,
        system_prompt=get_orchestrator_prompt(),
        model_name=model_name,
        max_loops=max_loops,
        max_tokens=8000,
        tools=tools,
        tool_call_summary=True,
        *args,
        **kwargs,
    )

    output = director_agent.run(task=task, img=img)
    TokenTracker.track(
        str(task) + director_agent.system_prompt, output, agent_name
    )
    return output


class AdvancedResearch:
    """
    AdvancedResearch is a high-level orchestrator for multi-agent research workflows.
    It manages the research process by coordinating director and worker agents, maintaining
    conversation history, and supporting export and output formatting.

    Attributes:
        id (str): Unique identifier for the research session.
        name (str): Name of the research system or session.
        description (str): Description of the research system or session.
        worker_model_name (str): Model name used for worker agents.
        director_agent_name (str): Name of the director agent.
        director_model_name (str): Model name used for the director agent.
        director_max_tokens (int): Maximum tokens for the director agent's output.
        output_type (HistoryOutputType): Output format for conversation history.
        max_loops (int): Number of research loops to run.
        export_on (bool): Whether to export conversation history to a JSON file.
        director_max_loops (int): Maximum loops for the director agent.
        conversation (Conversation): Conversation object to store the research dialogue.
    """

    def __init__(
        self,
        id: str = generate_id(),
        name: str = "Advanced Research",
        description: str = "Advanced Research",
        worker_model_name: str = model_name,
        director_agent_name: str = "Director-Agent",
        director_model_name: str = director_model_name,
        director_max_tokens: int = 8000,
        output_type: HistoryOutputType = "final",
        max_loops: int = 1,
        export_on: bool = False,
        director_max_loops: int = 1,
    ):
        """
        Initialize the AdvancedResearch system.

        Args:
            id (str): Unique identifier for the research session.
            name (str): Name of the research system or session.
            description (str): Description of the research system or session.
            worker_model_name (str): Model name for worker agents.
            director_agent_name (str): Name of the director agent.
            director_model_name (str): Model name for the director agent.
            director_max_tokens (int): Maximum tokens for the director agent's output.
            output_type (HistoryOutputType): Output format for conversation history.
            max_loops (int): Number of research loops to run.
            export_on (bool): Whether to export conversation history to a JSON file.
            director_max_loops (int): Maximum loops for the director agent.
        """
        self.id = id
        self.name = name
        self.description = description
        self.worker_model_name = worker_model_name
        self.director_agent_name = director_agent_name
        self.director_model_name = director_model_name
        self.director_max_tokens = director_max_tokens
        self.output_type = output_type
        self.max_loops = max_loops
        self.export_on = export_on
        self.director_max_loops = director_max_loops
        self.memory = ResearchMemory()

        self.conversation = Conversation(
            name=f"conversation-{self.id}"
        )

    def update_research_state(self, plan: str, findings: str) -> str:
        """
        Updates the research memory with a new plan and/or new findings.

        Args:
            plan (str): The updated research plan.
            findings (str): New key findings to add to the list.

        Returns:
            str: Confirmation message.
        """
        if plan:
            self.memory.plan = plan
        if findings:
            self.memory.findings.append(findings)
        return "Research state updated successfully."

    def step(self, task: Optional[str], img: Optional[str] = None):
        """
        Execute a single research step by running the director agent on the given task.

        Args:
            task (Optional[str]): The research task to execute.
            img (Optional[str]): Optional image input.

        Returns:
            str: The output from the director agent.
        """
        # Inject memory context into task
        memory_context = f"""
        Current Research Plan:
        {self.memory.plan}
        
        Key Findings So Far:
        {chr(10).join(f"- {f}" for f in self.memory.findings)}
        """

        full_task = (
            f"{task}\n\n[SYSTEM: MEMORY CONTEXT]\n{memory_context}"
        )

        # Run the director agent
        output = create_director_agent(
            agent_name=self.director_agent_name,
            model_name=self.director_model_name,
            task=full_task,
            img=img,
            extra_tools=[self.update_research_state],
        )

        if schema.enable_citation:
            logger.info("Running Citation Agent...")
            output = generate_citations(output)

        if schema.enable_evaluation:
            logger.info(
                "Running Evaluator Agent on Director Output..."
            )
            evaluation = evaluate_research(task, output)
            output = (
                f"{output}\n\n--- Final Evaluation ---\n{evaluation}"
            )

        self.conversation.add(self.director_agent_name, output)

        return output

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        **kwargs,
    ):
        """
        Run the advanced research system. Runs the research system for the specified number of loops,
        maintaining conversation history across all iterations.

        Args:
            task (str, optional): The research task to execute.
            img (Optional[str]): Optional image input.

        Returns:
            str or list: Formatted conversation history containing all loop iterations,
                         or exports the conversation to a JSON file if export_on is True.
                         Returns None when launching chat interface.
        """
        if task is None:
            raise ValueError(
                "task argument is required to run AdvancedResearch"
            )

        # Add the initial human task to conversation
        self.conversation.add("human", task)

        # Run the research system for the specified number of loops
        for loop_num in range(self.max_loops):
            logger.info(
                f"Starting research loop {loop_num + 1}/{self.max_loops}"
            )

            # Create a context-aware task that includes previous conversation history
            if loop_num == 0:
                # First loop: use the original task
                current_task = task
            else:
                # Subsequent loops: create a continuation task that references previous findings
                previous_output = (
                    self.conversation.get_final_message()
                )
                current_task = f"""
                Continue the research based on previous findings. 
                
                Original task: {task}
                
                Previous research findings: {previous_output}
                
                Please build upon the previous research and provide additional insights, 
                clarifications, or deeper analysis as needed.
                """

            # Execute the research step
            self.step(current_task, img)

            logger.info(
                f"Completed research loop {loop_num + 1}/{self.max_loops}"
            )

            # If export is enabled, save after each loop
            if self.export_on:
                self._export_conversation()

        # Log total token usage
        stats = TokenTracker.get_stats()
        logger.info(
            f"Research Session Complete. Total Token Usage: {stats}"
        )

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def batched_run(self, tasks: List[str]):
        """
        Run the research system on a batch of tasks.

        Args:
            tasks (List[str]): List of research tasks to execute.
        """
        [self.run(task) for task in tasks]

    def _export_conversation(self):
        """
        Export the conversation history to a JSON file.
        """
        if self.export_on:
            output_file = f"examples/outputs/{self.id}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            conversation_data = {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "conversation_history": self.conversation.get_history(),
                "export_timestamp": datetime.now().isoformat(),
            }

            create_json_file(conversation_data, output_file)
            logger.info(f"Conversation exported to {output_file}")

    def get_output_methods(self):
        """
        Get the available output formatting methods.

        Returns:
            list: List of available HistoryOutputType values.
        """
        return list(HistoryOutputType)
