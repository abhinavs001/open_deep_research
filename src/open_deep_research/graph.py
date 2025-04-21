from typing import Literal
import difflib

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command, Send
from open_deep_research_extension.retrieval.chroma_vector_store import ChromaVectorStore
from open_deep_research_extension.retrieval.reranker import rerank_documents
from typing import Literal, List, Dict, Any, Optional, Union


from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

## Nodes -- 

from langgraph.types import Command

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the initial report plan with sections, using streaming and feedback."""

    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)

    configurable = Configuration.from_runnable_config(config)
    report_structure = str(configurable.report_structure or "")
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}
    params_to_pass = get_search_params(search_api, search_api_config)

    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider)
    structured_llm = writer_model.with_structured_output(Queries)

    # Generate search queries
    query_instructions = report_planner_query_writer_instructions.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries
    )

    queries = structured_llm.invoke([
        SystemMessage(content=query_instructions),
        HumanMessage(content="Generate search queries that will help with planning the sections of the report.")
    ])

    query_list = [q.search_query for q in queries.queries]
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    # Handle feedback
    feedback_text = (
        f"The user provided the following feedback on the previous plan. You must strictly follow this feedback:\n\"{feedback}\""
        if feedback else
        "The user provided no feedback. Proceed normally."
    )

    # Generate section planning prompt
    section_instructions = report_planner_instructions.format(
        topic=topic,
        report_organization=report_structure,
        context=source_str,
        feedback=feedback_text
    )

    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    if planner_model == "claude-3-7-sonnet-latest":
        planner_llm = init_chat_model(
            model=planner_model,
            model_provider=planner_provider,
            max_tokens=20000,
            thinking={"type": "enabled", "budget_tokens": 16000}
        )
    else:
        planner_llm = init_chat_model(model=planner_model, model_provider=planner_provider)

    structured_planner = planner_llm.with_structured_output(Sections)

    # If you're using streaming, integrate this into your UI loop
    # Placeholder stream call (optional)
    # async for event in graph.astream(Command(resume=feedback_text), thread, stream_mode="updates"):
    #     print(event)

    report_sections = structured_planner.invoke([
        SystemMessage(content=section_instructions),
        HumanMessage(content="""Generate the sections of the report. 
            Your response must include a 'sections' field containing a list of sections. 
            Each section must have: name, description, plan, research, and content fields.""")
    ])

    return {"sections": report_sections.sections}

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan", "build_section_with_web_research"]]:
    """Get human feedback on the report plan and route to next steps."""

    topic = state["topic"]
    sections = state["sections"]

    # Format current sections
    sections_str = "\n\n".join(
        f"📌 Section: {s.name}\n📝 Description: {s.description}\n🔍 Needs Research: {'Yes' if s.research else 'No'}"
        for s in sections
    )

    # Prompt user
    prompt = f"""Please review the proposed report plan for the topic: {topic}
    {sections_str} Type 'true' if you approve the plan.
    Or, provide suggestions for improvement (e.g. "Include ARR estimates", "Split this into two parts", etc).
    """.strip()

    feedback = interrupt(prompt)

    # Handle approval
    if feedback is True or (isinstance(feedback, str) and feedback.strip().lower() == "true"):
        research_sections = [s for s in sections if s.research]
        if not research_sections:
            return Command(goto="retrieve_local_documents")
        return Command(goto=[
            Send("build_section_with_web_research", {
                "topic": topic,
                "section": s,
                "search_iterations": 0
            }) for s in research_sections
        ])

    # Handle feedback string
    elif isinstance(feedback, str):
        return Command(goto="generate_report_plan", update={"feedback_on_report_plan": feedback})

    raise TypeError(f"Unsupported feedback type: {type(feedback)}")

    
def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for researching a specific section.
    
    This node uses an LLM to generate targeted search queries based on the 
    section topic and description.
    
    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(topic=topic, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries)

    # Generate queries  
    queries = structured_llm.invoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

def write_section(state: SectionState, config: RunnableConfig) -> Command[str]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Format system instructions
    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 

    section_content = writer_model.invoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Write content to the section object  
    section.content = section_content.content

    # Grade prompt 
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, 
                                           max_tokens=20_000, 
                                           thinking={"type": "enabled", "budget_tokens": 16_000}).with_structured_output(Feedback)
    else:
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider).with_structured_output(Feedback)
    # Generate feedback
    feedback = reflection_model.invoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections 
        return  Command(
        update={"completed_sections": [section]},
        goto=END
    )

    # Update the existing section with new content and update search queries
    else:
        return  Command(
        update={"search_queries": feedback.follow_up_queries, "section": section},
        goto="search_web"
        )
    
def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.
    
    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.
    
    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model
        
    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get state 
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    
    section_content = writer_model.invoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """Format completed sections as context for writing final sections.
    
    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.
    
    Args:
        state: Current state with completed sections
        
    Returns:
        Dict with formatted sections as context
    """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}
import difflib

def compile_final_report(state: ReportState):
    """
    Compile all sections into the final report.
    
    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report
    
    Args:
        state: Current state with all completed sections
        
    Returns:
        Dict containing the complete report

    This function:
    1. Matches planned sections to completed ones using exact or fuzzy matching.
    2. Adds unmatched completed sections at the end to avoid loss of content.

    Args:
        state (ReportState): The current state containing 'sections' and 'completed_sections'.

    Returns:
        dict: A dictionary with the final compiled report under the key 'final_report'.
    """

    sections = state["sections"]
    completed_sections_raw = state["completed_sections"]

    # Build a normalized lookup of completed sections by name
    completed_lookup = {
        s.name.strip().lower(): s for s in completed_sections_raw
    }

    used_keys = set()
    compiled = []

    for section in sections:
        key = section.name.strip().lower()
        matched_section = completed_lookup.get(key)

        # Fuzzy match if exact key not found
        if not matched_section:
            close_matches = difflib.get_close_matches(
                key, completed_lookup.keys(), n=2, cutoff=0.6
            )
            if close_matches:
                matched_section = completed_lookup[close_matches[0]]

        if matched_section:
            used_keys.add(matched_section.name.strip().lower())
            section.content = matched_section.content
            compiled.append(section.content)
        else:
            print(f"⚠️ Warning: Missing completed content for: {section.name}")

    # Add any unmatched generated sections at the end
    for key, sec in completed_lookup.items():
        if key not in used_keys:
            print(f"➕ Including unmatched section: {sec.name}")
            compiled.append(sec.content)

    return {"final_report": "\n\n".join(compiled)}

def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.
    
    Args:
        state: Current state with all sections and research context
        
    Returns:
        List of Send commands for parallel section writing
    """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]

def retrieve_local_documents(state: ReportState, config: RunnableConfig):
    topic = state["topic"]
    vectordb = ChromaVectorStore().load()
    retriever = vectordb.as_retriever()
    documents = retriever.invoke(topic)
    reranked = rerank_documents(topic, documents)
    return {"source_str": "\n\n".join([doc.page_content for doc in reranked]), "search_iterations": 0}

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("retrieve_local_documents", retrieve_local_documents)
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_edge("generate_report_plan", "retrieve_local_documents")
builder.add_edge("retrieve_local_documents", "build_section_with_web_research")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile(interrupt_before=["human_feedback"])
