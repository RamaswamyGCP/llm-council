"""LangGraph workflow for the LLM Council."""

from typing import List, Dict, Any, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL
from .openrouter import query_models_parallel, query_model
from .utils import parse_ranking_from_text, calculate_aggregate_rankings

class CouncilState(TypedDict):
    """State for the Council workflow."""
    user_query: str
    stage1_results: List[Dict[str, Any]]
    stage2_results: List[Dict[str, Any]]
    label_to_model: Dict[str, str]
    stage3_result: Dict[str, Any]
    metadata: Dict[str, Any]

async def stage1_node(state: CouncilState) -> Dict[str, Any]:
    """Stage 1: Collect individual responses."""
    user_query = state["user_query"]
    messages = [{"role": "user", "content": user_query}]
    
    # Query all models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)
    
    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })
            
    return {"stage1_results": stage1_results}

async def stage2_node(state: CouncilState) -> Dict[str, Any]:
    """Stage 2: Rank responses."""
    user_query = state["user_query"]
    stage1_results = state["stage1_results"]
    
    if not stage1_results:
        return {
            "stage2_results": [], 
            "label_to_model": {}, 
            "metadata": {"error": "No stage 1 results"}
        }

    # Create anonymized labels
    labels = [chr(65 + i) for i in range(len(stage1_results))]
    
    # Map labels to models
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }
    
    # Build ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])
    
    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually.
2. Then, at the very end, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with "FINAL RANKING:"
- List responses from best to worst (e.g., "1. Response A")

FINAL RANKING:
1. Response [Label]
2. Response [Label]
...
"""

    messages = [{"role": "user", "content": ranking_prompt}]
    
    # Get rankings from all models
    responses = await query_models_parallel(COUNCIL_MODELS, messages)
    
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })
            
    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
    
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }
    
    return {
        "stage2_results": stage2_results,
        "label_to_model": label_to_model,
        "metadata": metadata
    }

async def stage3_node(state: CouncilState) -> Dict[str, Any]:
    """Stage 3: Synthesize final answer."""
    user_query = state["user_query"]
    stage1_results = state["stage1_results"]
    stage2_results = state["stage2_results"]
    
    if not stage1_results:
        return {"stage3_result": {"model": "error", "response": "No responses to synthesize."}}

    # Build context
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Synthesize all information into a single, comprehensive answer:"""

    messages = [{"role": "user", "content": chairman_prompt}]
    
    # Query chairman
    response = await query_model(CHAIRMAN_MODEL, messages)
    
    if response is None:
        return {"stage3_result": {"model": CHAIRMAN_MODEL, "response": "Error: Synthesis failed."}}
        
    return {
        "stage3_result": {
            "model": CHAIRMAN_MODEL,
            "response": response.get('content', '')
        }
    }

def create_council_graph():
    """Create the LangGraph workflow."""
    workflow = StateGraph(CouncilState)
    
    workflow.add_node("stage1", stage1_node)
    workflow.add_node("stage2", stage2_node)
    workflow.add_node("stage3", stage3_node)
    
    workflow.set_entry_point("stage1")
    workflow.add_edge("stage1", "stage2")
    workflow.add_edge("stage2", "stage3")
    workflow.add_edge("stage3", END)
    
    return workflow.compile()
