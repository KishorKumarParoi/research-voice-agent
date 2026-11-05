from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CitationSource(BaseModel):
    id: str
    title: str
    authors: str
    year: int
    venue: str # arXiv, IEEE Trans, ACM SIGCOMM, Nature AI
    doi: str
    key_finding: str
    confidence_score: float # 0.0 - 1.0

class DecompositionNode(BaseModel):
    step_num: int
    sub_query: str
    evidence_found: str
    status: str = "verified" # verified, analyzing, pending

class LatencyMetrics(BaseModel):
    retrieval_ms: float = 0.0
    synthesis_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0

class ResearchMemo(BaseModel):
    id: str
    query: str
    executive_summary: str
    key_findings: List[str]
    citations: List[CitationSource]
    decomposition_tree: List[DecompositionNode]
    latex_formulation: str
    audio_base64: Optional[str] = None
    latency: LatencyMetrics

class InquireRequest(BaseModel):
    query: str
    depth: str = "deep_literature" # quick_brief, deep_literature, benchmark
    tts_provider: str = "elevenlabs"

class InquireResponse(BaseModel):
    memo: ResearchMemo
    spoken_reply: str
