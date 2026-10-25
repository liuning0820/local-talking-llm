"""Evidence-first prompts used by the local summary adapter."""

PROMPT_VERSION = "meeting-summary-v3"
SYSTEM_PROMPT = """You create a factual meeting record from only the supplied transcript.
Return one JSON object and no Markdown. Never invent names, dates, decisions,
topics, or actions. Use exact transcript segment IDs as evidence_segment_ids.
A factual decision or action must include a non-empty evidence_segment_ids list,
start_ms, and end_ms copied from the evidence. If the transcript does not state
a fact, use the exact text 未提及. Owner and due_date must be 未提及 when absent.
Every inference must contain both the labels 模型推断 and 需确认, and must not be
presented as a decision or action. Include metadata: {source: transcript,
version: 1}."""


def summary_prompt(transcript: str, purpose: str = "final") -> str:
    return f"""{SYSTEM_PROMPT}
Purpose: {purpose}
Required JSON keys: metadata, summary, topics, decisions, action_items,
unresolved_questions, inferences, missing_information. Every listed field must
be a JSON array, including summary; put one string in the array when there is
only one item. For absent or unknown content, use ["未提及"]. Never return a
scalar, null, or object in place of one of these arrays.
Each factual decision and action item must include evidence_segment_ids,
start_ms/end_ms, and action items must include owner, due_date, and status.
Transcript:
{transcript}
"""
