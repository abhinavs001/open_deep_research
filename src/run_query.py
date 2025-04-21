import asyncio
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv
from open_deep_research.graph import builder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
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
RESEARCH_TOPIC = "Compare the R&D expenditure trends across Apple, Amazon, and Microsoft in Q1 2023. What strategic differences are reflected in their investment priorities?"

# Normalize section names for consistent lookup
def normalize_section_key(name: str) -> str:
    return name.strip().lower()

async def main():
    # Check environment variables
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY", "ANTHROPIC_API_KEY", 
                    "LINKUP_API_KEY", "EXA_API_KEY", "GOOGLE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        print(f"Missing environment variables: {', '.join(missing_keys)}")
        return

    # Set up memory saver and graph
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    # Initialize the graph with the research topic
    print("\n--- Phase 1: Planning ---")
    start_state = {"topic": RESEARCH_TOPIC}
    
    # Start the graph and get to the planning stage
    async for event in graph.astream(start_state, THREAD_CONFIG, stream_mode="values"):
        if "sections" in event:
            print("Report plan generated!")
            break
    
    # Get the sections and display them
    sections = event.get("sections", [])
    topic = event.get("topic")
    
    print("\nGenerated Report Plan:")
    sections_str = "\n\n".join(
        f"Section {i+1}: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for i, section in enumerate(sections)
    )
    print(sections_str)
    
    # Get feedback from user
    print("\n===== HUMAN FEEDBACK REQUIRED =====")
    print("The graph will stop when the report plan is generated, and you can pass feedback to update the report plan.")
    feedback = input("\nDoes this report plan meet your needs? (type 'true' to approve or provide feedback): ")
    
    # Continue with execution based on feedback
    if feedback.lower() == 'true':
        print("Plan approved! Proceeding with research...")
        plan_approved = True
    else:
        print(f"Feedback received: '{feedback}'")
        print("Regenerating report plan based on feedback...")
        
        # Use Command(resume=feedback) to send feedback and regenerate the plan
        async for event in graph.astream(Command(resume=feedback, goto="generate_report_plan", update={"feedback_on_report_plan": feedback}), 
                                        THREAD_CONFIG, stream_mode="values"):
            if "sections" in event:
                print("Updated report plan generated!")
                sections = event.get("sections", [])
                
                print("\nUpdated Report Plan:")
                sections_str = "\n\n".join(
                    f"Section {i+1}: {section.name}\n"
                    f"Description: {section.description}\n"
                    f"Research needed: {'Yes' if section.research else 'No'}\n"
                    for i, section in enumerate(sections)
                )
                print(sections_str)
                
                # Ask if the updated plan is acceptable
                accept_plan = input("\nIs this updated plan acceptable? (yes/no): ")
                if accept_plan.lower() in ['yes', 'y', 'true']:
                    plan_approved = True
                    print("Updated plan approved! Proceeding with research...")
                else:
                    print("Process terminated. Please run again with different feedback.")
                    return
                break
    
    # If plan is approved, continue with the research and writing phases
    if plan_approved:
        # Dictionary to track completed sections
        completed_sections = {}
        
        # Phase 2: Research and write sections
        print("\n--- Phase 2: Writing Research Sections ---")
        research_sections = [s for s in sections if s.research]
        
        for section in research_sections:
            print(f"\nWorking on research section: {section.name}")
            try:
                cmd = Send("build_section_with_web_research", {
                    "topic": topic,
                    "section": section,
                    "search_iterations": 0
                })
                async for event in graph.astream(cmd, THREAD_CONFIG, stream_mode="values"):
                    if "completed_sections" in event:
                        for s in event["completed_sections"]:
                            completed_sections[normalize_section_key(s.name)] = s
                            print(f"Completed section: {s.name}")
            except Exception as e:
                print(f"Error processing section '{section.name}': {e}")
        
        # Gather completed sections for context
        if completed_sections:
            print("\n--- Gathering Completed Sections ---")
            cmd = Send("gather_completed_sections", {
                "topic": topic,
                "sections": sections,
                "completed_sections": list(completed_sections.values())
            })
            
            async for event in graph.astream(cmd, THREAD_CONFIG, stream_mode="values"):
                if "report_sections_from_research" in event:
                    report_sections_from_research = event["report_sections_from_research"]
                    print("Research sections gathered for context")
                    break
        else:
            report_sections_from_research = ""
            print("No research sections to gather")
        
        # Phase 3: Write non-research sections
        print("\n--- Phase 3: Writing Final Sections ---")
        final_sections = [s for s in sections if not s.research]
        
        for section in final_sections:
            print(f"\nWorking on final section: {section.name}")
            try:
                cmd = Send("write_final_sections", {
                    "topic": topic,
                    "section": section,
                    "report_sections_from_research": report_sections_from_research
                })
                
                async for event in graph.astream(cmd, THREAD_CONFIG, stream_mode="values"):
                    if "completed_sections" in event:
                        for s in event["completed_sections"]:
                            completed_sections[normalize_section_key(s.name)] = s
                            print(f"Completed section: {s.name}")
            except Exception as e:
                print(f"Error processing section '{section.name}': {e}")
        
        # Compile final report
        print("\n--- Final Compilation ---")
        try:
            cmd = Send("compile_final_report", {
                "topic": topic,
                "sections": sections,
                "completed_sections": list(completed_sections.values())
            })
            
            async for event in graph.astream(cmd, THREAD_CONFIG, stream_mode="values"):
                if "final_report" in event:
                    final_report = event["final_report"]
                    break
                
            # Output Final Report
            print("\n--- Final Output ---")
            if final_report:
                print("\n📄 Final Report:\n" + "="*40)
                print(final_report)
                print("="*40)
                
                # Save the report to a file
                #with open("final_report.md", "w") as f:
                    #f.write(final_report)
                #print(f"Report saved to final_report.md")
            else:
                print("No final report generated.")
        except Exception as e:
            print(f"Error during compilation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())