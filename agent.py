import os
import asyncio
from typing import TypedDict, Literal, List, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

# Load env early
load_dotenv()

class ResearchState(TypedDict):
    topic: str
    raw_data: str
    draft_summary: str
    final_report: str
    revision_feedback: str
    iteration_count: int

async def researcher_node(state: ResearchState) -> ResearchState:
    topic = state.get("topic", "")
    iteration_count = state.get("iteration_count", 0)
    feedback = state.get("revision_feedback", "")
    existing_data = state.get("raw_data", "")
    
    query = topic
    if iteration_count > 0 and feedback:
        query = f"{topic} {feedback}"
        
    search_tool = TavilySearchResults(max_results=4)
    # Perform sync call since simple tool wrapper isn't natively async sometimes
    results = search_tool.invoke({"query": query})
    
    if isinstance(results, list):
        formatted_data = "\n\n".join([f"Source: {res.get('url', 'N/A')}\nContent: {res.get('content', 'N/A')}" for res in results])
    else:
        formatted_data = str(results)
        
    new_data = f"--- Search Results Iteration {iteration_count + 1} ---\n{formatted_data}"
    combined_data = existing_data + "\n\n" + new_data if existing_data else new_data
    
    return {"raw_data": combined_data}

async def analyst_node(state: ResearchState) -> ResearchState:
    topic = state.get("topic", "")
    raw_data = state.get("raw_data", "")
    
    # We use versatile model as decided in prior fixes
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    
    prompt = f"""
    You are an expert analyst. Your task is to write a well-formatted, comprehensive 
    draft summary based on the provided raw research data.
    
    Topic: {topic}
    
    Raw Data:
    {raw_data}
    
    Please synthesize this information into a structured draft (Markdown format).
    """
    response = await llm.ainvoke(prompt)
    return {"draft_summary": response.content}

async def reviewer_node(state: ResearchState) -> ResearchState:
    topic = state.get("topic", "")
    raw_data = state.get("raw_data", "")
    draft_summary = state.get("draft_summary", "")
    iteration_count = state.get("iteration_count", 0)
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    
    prompt = f"""
    You are a meticulous reviewer and fact-checker. Review the draft summary against the raw data.
    
    If the draft is sufficient and has no hallucinations, reply with:
    [APPROVED]
    (followed by the final polished report in markdown)
    
    If the raw data is insufficient to fully answer the topic, or important details are missing, reply with:
    [REJECTED]
    (followed by a single sentence describing exactly what additional information is needed)
    
    Topic: {topic}
    Raw Data: {raw_data}
    Draft Summary: {draft_summary}
    """
    response = await llm.ainvoke(prompt)
    content = response.content.strip()
    
    if content.startswith("[REJECTED]"):
        feedback = content.replace("[REJECTED]", "").strip()
        return {"revision_feedback": feedback, "iteration_count": iteration_count + 1}
    else:
        final_text = content.replace("[APPROVED]", "").strip()
        return {"final_report": final_text, "iteration_count": iteration_count + 1, "revision_feedback": ""}

def router(state: ResearchState) -> Literal["researcher", "__end__"]:
    # Loop back to researcher if feedback is provided, max 3 loops to avoid infinite loops
    if state.get("revision_feedback") and state.get("iteration_count", 0) < 3:
        return "researcher"
    return "__end__"

def build_async_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("reviewer", reviewer_node)
    
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "reviewer")
    
    # Conditional Edge
    builder.add_conditional_edges("reviewer", router)
    
    return builder.compile()

# Global compiled graph
app = build_async_graph()
