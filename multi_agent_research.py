import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables from .env file
load_dotenv()

# ------------------------------------------------------------------------------
# 1. State Definition
# ------------------------------------------------------------------------------
class ResearchState(TypedDict):
    """
    Represents the state of our multi-agent research workflow.
    """
    topic: str
    raw_data: str
    draft_summary: str
    final_report: str

# ------------------------------------------------------------------------------
# 2. Node Functions
# ------------------------------------------------------------------------------
def researcher_node(state: ResearchState) -> ResearchState:
    """
    Uses the Tavily search tool to browse the web for recent information
    based on the topic, and saves the findings to `raw_data`.
    """
    print("-> Running [Researcher Node]...")
    topic = state.get("topic", "")
    
    try:
        tagline = f"Latest findings on {topic}"
        # Initialize the Tavily tool
        search_tool = TavilySearchResults(max_results=4)
        
        # Invoke search
        results = search_tool.invoke({"query": topic})
        
        # Format the results into a string block
        if isinstance(results, list):
            formatted_data = "\n\n".join(
                [f"Source: {res.get('url', 'N/A')}\nContent: {res.get('content', 'N/A')}" 
                 for res in results]
            )
        else:
            # Fallback if the results aren't a list
            formatted_data = str(results)
            
        return {"raw_data": f"--- Search Results ---\n{formatted_data}"}
        
    except Exception as e:
        error_msg = f"Error during web search: {str(e)}"
        print(f"   [Error] {error_msg}")
        return {"raw_data": error_msg}


def analyst_node(state: ResearchState) -> ResearchState:
    """
    Takes the `raw_data` and uses the Groq LLM to synthesize the 
    information into a structured, well-formatted draft summary.
    """
    print("-> Running [Analyst Node]...")
    topic = state.get("topic", "")
    raw_data = state.get("raw_data", "")
    
    try:
        # Initialize Groq LLM
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        
        prompt = f"""
        You are an expert analyst. Your task is to write a well-formatted, comprehensive 
        draft summary based on the provided raw research data.
        
        Topic: {topic}
        
        Raw Data:
        {raw_data}
        
        Please synthesize this information into a structured draft.
        """
        
        # Invoke LLM
        response = llm.invoke(prompt)
        
        # The content of the response represents our draft
        return {"draft_summary": response.content}
        
    except Exception as e:
        error_msg = f"Error generating draft: {str(e)}"
        print(f"   [Error] {error_msg}")
        return {"draft_summary": error_msg}


def reviewer_node(state: ResearchState) -> ResearchState:
    """
    Takes the `draft_summary` and `raw_data`, uses the Groq LLM to fact-check 
    the draft against the raw data for hallucinations, and outputs the final report.
    """
    print("-> Running [Reviewer Node]...")
    topic = state.get("topic", "")
    raw_data = state.get("raw_data", "")
    draft_summary = state.get("draft_summary", "")
    
    try:
        # Initialize Groq LLM
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        
        prompt = f"""
        You are a meticulous reviewer and fact-checker. Your task is to review 
        the draft summary below against the provided raw data.
        
        Goal: 
        1. Ensure the draft does not contain any hallucinations or claims NOT supported by the raw data.
        2. Correct any inaccuracies.
        3. Output the final, polished report in a clear, highly-readable format.
        
        Topic: {topic}
        
        --- RAW DATA ---
        {raw_data}
        
        --- DRAFT SUMMARY ---
        {draft_summary}
        
        Output ONLY the final, refined report. Do not include any extra conversational filler.
        """
        
        # Invoke LLM
        response = llm.invoke(prompt)
        
        return {"final_report": response.content}
        
    except Exception as e:
        error_msg = f"Error during review: {str(e)}"
        print(f"   [Error] {error_msg}")
        return {"final_report": error_msg}


# ------------------------------------------------------------------------------
# 3. Graph Construction
# ------------------------------------------------------------------------------
def build_research_graph():
    """
    Builds and compiles the StateGraph workflow.
    """
    # Create the graph builder with our TypedDict state
    builder = StateGraph(ResearchState)
    
    # Add all three nodes
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("reviewer", reviewer_node)
    
    # Define the sequential flow
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "reviewer")
    builder.add_edge("reviewer", END)
    
    # Compile the graph into a runnable workflow
    return builder.compile()


# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if necessary API keys are present in the environment
    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("⚠️ Missing API Keys!")
        print("Please ensure both GROQ_API_KEY and TAVILY_API_KEY are set in your .env file or environment.")
        exit(1)
        
    # Build the computational graph
    app = build_research_graph()
    
    # Define our initial state with the requested topic
    initial_topic = "Latest advancements in Agentic AI workflows"
    
    initial_state = {
        "topic": initial_topic,
        "raw_data": "",
        "draft_summary": "",
        "final_report": ""
    }
    
    print("=" * 60)
    print(f"STARTING RESEARCH WORKFLOW")
    print(f"Topic: {initial_topic}")
    print("=" * 60)
    
    try:
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        # Output the Final Report
        print("\n\n" + "=" * 60)
        print(" " * 20 + "FINAL REPORT")
        print("=" * 60 + "\n")
        print(final_state.get("final_report", "No report generated."))
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n[Fatal Error] Workflow execution failed: {e}")
