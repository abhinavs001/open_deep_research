import asyncio
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv
from open_deep_research.graph import builder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Send

# Load environment variables from 'env' file
load_dotenv(dotenv_path=Path("env"))

# Define thread config
THREAD_CONFIG = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
        "search_api": "tavily",
        "planner_provider": "openai",
        "planner_model": "o3-mini",
        "writer_provider": "openai",
        "writer_model": "o3-mini",
        "max_search_depth": 1,
    }
}

# Topic to generate report for
RESEARCH_TOPIC = "Summarize and compare the agency priority goals vs cross-agency priority goals as defined in OPM's 2022 strategy."

# Normalize section names for consistent lookup
def normalize_section_key(name: str) -> str:
    return name.strip().lower()

async def main():
    # Check environment variables
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY","LANGSMITH_API_KEY","ANTHROPIC_API_KEY","LINKUP_API_KEY","EXA_API_KEY","GOOGLE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        print(f"Missing environment variables: {', '.join(missing_keys)}")
        return

    # Compile graph
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    final_report = None
    last_state = None
    completed_sections = {}

    # Phase 1: Planning
    print("\n--- Phase 1: Planning ---")
    async for event in graph.astream({"topic": RESEARCH_TOPIC}, THREAD_CONFIG, stream_mode="values"):
        last_state = event

    sections = last_state.get("sections", [])
    topic = last_state.get("topic")

    # Phase 2: Writing Research Sections
    print("\n--- Phase 2: Writing Research Sections ---")
    for section in sections:
        if section.research:
            async for event in graph.astream(
                Send("build_section_with_web_research", {
                    "topic": topic,
                    "section": section,
                    "search_iterations": 0
                }),
                THREAD_CONFIG,
                stream_mode="values"
            ):
                last_state = event
                if "completed_sections" in event:
                    for s in event["completed_sections"]:
                        completed_sections[normalize_section_key(s.name)] = s

    # Phase 3: Writing Final Sections
    print("\n--- Phase 3: Writing Final Sections ---")
    for section in sections:
        if not section.research:
            async for event in graph.astream(
                Send("write_final_sections", {
                    "topic": topic,
                    "section": section,
                    "report_sections_from_research": last_state.get("report_sections_from_research", "")
                }),
                THREAD_CONFIG,
                stream_mode="values"
            ):
                last_state = event
                if "completed_sections" in event:
                    for s in event["completed_sections"]:
                        completed_sections[normalize_section_key(s.name)] = s

    # Store final list of sections into last_state
    last_state["completed_sections"] = list(completed_sections.values())

    # Final Compilation
    print("\n--- Final Compilation ---")
    async for event in graph.astream(Command(resume=True), THREAD_CONFIG, stream_mode="values"):
        last_state = event
        if "final_report" in event:
            final_report = event["final_report"]
            break

    # Output Final Report
    print("\n--- Final Output ---")
    if final_report:
        print("\n📄 Final Report:\n" + "="*40)
        print(final_report)
        print("="*40)
    else:
        print("No final report generated.")
        if last_state:
            print("\nLast known state keys:", list(last_state.keys()))

if __name__ == "__main__":
    asyncio.run(main())
