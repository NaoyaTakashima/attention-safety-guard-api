"""
Attention Safety Guard - FastAPI Server
Optimized for RunPod Serverless deployment
"""

import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core import DEFAULT_POLICY, AnalysisResult, analyze_prompt, load_model


class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint"""

    prompt: str = Field(..., description="User prompt to analyze", min_length=1, max_length=10000)
    policy: str | None = Field(None, description="Custom safety policy (optional)")
    max_new_tokens: int = Field(256, description="Max tokens to generate", ge=64, le=512)


class TokenAttention(BaseModel):
    """Token with attention score"""

    token: str
    score: float


class AnalyzeResponse(BaseModel):
    """Response body for /analyze endpoint"""

    prompt: str
    verdict: Literal["Violating", "Non-violating", "Unknown"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    policy_attention_ratio: float = Field(..., ge=0.0, le=1.0)
    content_attention_ratio: float = Field(..., ge=0.0, le=1.0)
    top_tokens: list[TokenAttention]  # Top tokens from entire input
    top_content_tokens: list[TokenAttention]  # Top tokens from user content only
    all_token_scores: list[TokenAttention]  # All tokens with scores
    content_boundary_idx: int  # Token index where content starts
    generated_text: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response body for /health endpoint"""

    status: Literal["healthy", "loading", "error"]
    model_loaded: bool
    gpu_available: bool
    gpu_memory_gb: float | None


def result_to_response(result: AnalysisResult, processing_time_ms: float) -> AnalyzeResponse:
    """Convert AnalysisResult to API response"""
    return AnalyzeResponse(
        prompt=result.prompt,
        verdict=result.verdict,  # type: ignore[arg-type]
        confidence=result.confidence,
        reasoning=result.reasoning,
        policy_attention_ratio=result.policy_attention_ratio,
        content_attention_ratio=result.content_attention_ratio,
        top_tokens=[TokenAttention(token=t, score=s) for t, s in result.top_tokens],
        top_content_tokens=[TokenAttention(token=t, score=s) for t, s in result.top_content_tokens],
        all_token_scores=[TokenAttention(token=t, score=s) for t, s in result.all_token_scores],
        content_boundary_idx=result.content_boundary_idx,
        generated_text=result.generated_text,
        processing_time_ms=processing_time_ms,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model on startup for faster first request"""
    print("Starting model preload...")
    load_model()
    print("Model preload complete!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Attention Safety Guard API",
    description="gpt-oss-safeguard with Attention Rollout analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for RunPod"""
    import torch

    from core import _model

    gpu_available = torch.cuda.is_available()
    gpu_memory_gb = None
    if gpu_available:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    return HealthResponse(
        status="healthy" if _model is not None else "loading",
        model_loaded=_model is not None,
        gpu_available=gpu_available,
        gpu_memory_gb=gpu_memory_gb,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a prompt for safety violations using gpt-oss-safeguard.

    Returns verdict, confidence, attention analysis, and top influential tokens.
    """
    try:
        start_time = time.perf_counter()

        result = analyze_prompt(
            prompt=request.prompt,
            policy=request.policy,
            max_new_tokens=request.max_new_tokens,
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return result_to_response(result, processing_time_ms)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") from e


@app.get("/policy")
async def get_default_policy() -> dict[str, str]:
    """Get the default safety policy"""
    return {"policy": DEFAULT_POLICY}


# RunPod Serverless handler
def handler(event: dict) -> dict:
    """
    RunPod Serverless handler function.

    Expected input format:
    {
        "input": {
            "prompt": "...",
            "policy": "..." (optional),
            "max_new_tokens": 256 (optional)
        }
    }
    """
    try:
        input_data = event.get("input", {})

        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        policy = input_data.get("policy")
        max_new_tokens = input_data.get("max_new_tokens", 256)

        start_time = time.perf_counter()

        result = analyze_prompt(
            prompt=prompt,
            policy=policy,
            max_new_tokens=max_new_tokens,
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "prompt": result.prompt,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "policy_attention_ratio": result.policy_attention_ratio,
            "content_attention_ratio": result.content_attention_ratio,
            "top_tokens": [{"token": t, "score": s} for t, s in result.top_tokens],
            "top_content_tokens": [{"token": t, "score": s} for t, s in result.top_content_tokens],
            "all_token_scores": [{"token": t, "score": s} for t, s in result.all_token_scores],
            "content_boundary_idx": result.content_boundary_idx,
            "generated_text": result.generated_text,
            "processing_time_ms": processing_time_ms,
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
