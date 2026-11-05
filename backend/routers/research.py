from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from backend.models import InquireRequest, InquireResponse, CitationSource
from backend.research_engine import research_engine
from backend.core.tts import generate_speech
from backend.core.livekit import generate_livekit_token
from backend.config import settings

router = APIRouter(prefix="/api/research", tags=["Research Engine"])

class TokenReq(BaseModel):
    room_name: str
    participant_identity: str

@router.get("/corpus", response_model=List[CitationSource])
async def get_corpus():
    return research_engine.papers

@router.post("/inquire", response_model=InquireResponse)
async def perform_inquiry(req: InquireRequest):
    memo, spoken = await research_engine.execute_inquiry(req.query)
    audio_base64, tts_ms = await generate_speech(spoken, provider=req.tts_provider)
    memo.audio_base64 = audio_base64
    if tts_ms > 0:
        memo.latency.tts_ms = tts_ms
        memo.latency.total_ms = round(memo.latency.retrieval_ms + memo.latency.synthesis_ms + tts_ms, 2)

    return InquireResponse(memo=memo, spoken_reply=spoken)

@router.post("/token")
async def get_research_token(req: TokenReq):
    token = generate_livekit_token(req.room_name, req.participant_identity)
    return {
        "server_url": settings.LIVEKIT_URL,
        "token": token,
        "room_name": req.room_name,
        "participant_identity": req.participant_identity
    }

