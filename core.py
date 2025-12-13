"""
Attention Safety Guard - Core Logic
Shared between Gradio app and FastAPI server
"""

from dataclasses import dataclass
from pathlib import Path

import torch

# Global model cache
_model = None
_tokenizer = None


@dataclass
class AnalysisResult:
    """Analysis result data class"""

    prompt: str
    verdict: str
    confidence: float
    reasoning: str
    policy_attention_ratio: float
    content_attention_ratio: float
    top_tokens: list[tuple[str, float]]  # Top tokens from entire input
    top_content_tokens: list[tuple[str, float]]  # Top tokens from user content only
    all_token_scores: list[tuple[str, float]]  # All tokens with scores
    content_boundary_idx: int  # Token index where content starts
    generated_text: str


def load_model():
    """Load gpt-oss-safeguard model (cached)"""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "openai/gpt-oss-safeguard-20b"

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        # Use only the first visible GPU (respects CUDA_VISIBLE_DEVICES)
        device = torch.device("cuda:0")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {gpu_memory:.1f} GB")

        # Use device_map with explicit single GPU to avoid multi-GPU distribution
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},  # Load all layers to cuda:0
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            trust_remote_code=True,
        )

    _model = model
    _tokenizer = tokenizer

    print("Model loaded successfully!")
    return model, tokenizer


def get_default_policy() -> str:
    """Get default policy from file or fallback"""
    policy_path = Path(__file__).parent.parent / "data" / "policy.md"
    if policy_path.exists():
        return policy_path.read_text()

    return """# Safety Policy

You are a content safety classifier. Your task is to determine if user content violates any safety policies.

## Disallowed Content Categories

1. **Violence & Harm**: Content promoting violence, self-harm, or harm to others
2. **Hate Speech**: Discrimination or attacks based on protected characteristics
3. **Sexual Content**: Explicit sexual content or exploitation
4. **Illegal Activities**: Instructions for illegal activities, drugs, weapons
5. **Fraud & Deception**: Phishing, scams, impersonation
6. **Privacy Violations**: Sharing personal information without consent
7. **Malware & Hacking**: Code or instructions for malicious software

## Response Format

Analyze the content and respond with:
- Reasoning: Brief explanation of your analysis
- Answer: 1 if violating, 0 if safe
"""


DEFAULT_POLICY = get_default_policy()


def find_content_boundary(tokenizer, full_input: str, input_ids: torch.Tensor) -> int:
    """Find the boundary between policy and content tokens.

    Uses the LAST occurrence of 'Content:' to find where user input starts,
    since policy may contain examples with 'Content:' markers.
    """
    content_marker = "\n\nContent:"
    # Use rfind to get the LAST occurrence (user's actual input)
    char_pos = full_input.rfind(content_marker)

    if char_pos == -1:
        return len(input_ids[0]) // 2

    prefix_text = full_input[: char_pos + len(content_marker)]
    prefix_tokens = tokenizer(prefix_text, return_tensors="pt")
    return prefix_tokens["input_ids"].shape[1]


def calculate_attention_ratios(scores: list[float], boundary_idx: int) -> tuple[float, float]:
    """Calculate policy vs content attention ratios with length normalization"""
    if not scores or boundary_idx <= 0 or boundary_idx >= len(scores):
        return 0.5, 0.5

    total_tokens = len(scores)
    policy_tokens = boundary_idx
    content_tokens = total_tokens - boundary_idx

    if policy_tokens == 0 or content_tokens == 0:
        return 0.5, 0.5

    total_score = sum(scores)
    if total_score == 0:
        return 0.5, 0.5

    normalized_scores = [s / total_score for s in scores]

    policy_attention = sum(normalized_scores[:boundary_idx])
    content_attention = sum(normalized_scores[boundary_idx:])

    policy_expected = policy_tokens / total_tokens
    content_expected = content_tokens / total_tokens

    policy_normalized = policy_attention / policy_expected if policy_expected > 0 else 0
    content_normalized = content_attention / content_expected if content_expected > 0 else 0

    total_normalized = policy_normalized + content_normalized
    if total_normalized == 0:
        return 0.5, 0.5

    return policy_normalized / total_normalized, content_normalized / total_normalized


def extract_verdict_from_text(text: str) -> tuple[str, float]:
    """Extract verdict from generated text using keyword matching"""
    text_lower = text.lower()

    violating_keywords = ["violating", "violation", "answer: 1", "is a violation"]
    non_violating_keywords = ["non-violating", "not violating", "answer: 0", "not a violation"]

    has_violating = any(kw in text_lower for kw in violating_keywords)
    has_non_violating = any(kw in text_lower for kw in non_violating_keywords)

    if has_violating and not has_non_violating:
        return "Violating", 0.8
    elif has_non_violating:
        return "Non-violating", 0.8
    elif has_violating and has_non_violating:
        v_pos = max((text_lower.rfind(kw) for kw in violating_keywords if kw in text_lower), default=-1)
        nv_pos = max((text_lower.rfind(kw) for kw in non_violating_keywords if kw in text_lower), default=-1)
        return ("Violating" if v_pos > nv_pos else "Non-violating"), 0.6
    else:
        return "Unknown", 0.3


def analyze_prompt(
    prompt: str,
    policy: str | None = None,
    max_new_tokens: int = 256,
) -> AnalysisResult:
    """Analyze a prompt using gpt-oss-safeguard with attention analysis"""

    if policy is None:
        policy = DEFAULT_POLICY

    model, tokenizer = load_model()

    full_input = f"{policy}\n\nContent: {prompt}\nReasoning:"

    inputs = tokenizer(full_input, return_tensors="pt")
    input_ids = inputs["input_ids"]

    content_boundary_idx = find_content_boundary(tokenizer, full_input, input_ids)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

    attribution_scores = []

    with torch.no_grad():
        forward_outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

        if hasattr(forward_outputs, "attentions") and forward_outputs.attentions is not None:
            num_tokens = input_ids.shape[1]

            rollout = torch.eye(num_tokens)

            for layer_attention in forward_outputs.attentions:
                attn = layer_attention[0].mean(dim=0).cpu()
                attn_with_residual = 0.5 * attn + 0.5 * torch.eye(num_tokens)
                attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
                rollout = torch.matmul(rollout, attn_with_residual)

            attention_to_input = rollout[-1, :]
            scores_list = attention_to_input.tolist()

            if len(scores_list) > 1:
                avg_without_first = sum(scores_list[1:]) / len(scores_list[1:])
                scores_list[0] = avg_without_first

            attribution_scores = scores_list

        del forward_outputs
        torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0]
    generated_text = tokenizer.decode(generated_ids[input_ids.shape[1] :], skip_special_tokens=True)

    verdict, confidence = extract_verdict_from_text(generated_text)

    policy_ratio, content_ratio = calculate_attention_ratios(attribution_scores, content_boundary_idx)

    # All tokens with scores
    all_token_scores = list(zip(input_tokens, attribution_scores))

    # Top tokens from entire input
    top_tokens = sorted(all_token_scores, key=lambda x: abs(x[1]), reverse=True)[:10]

    # Top tokens from content only (user input)
    content_token_scores = all_token_scores[content_boundary_idx:]
    top_content_tokens = sorted(content_token_scores, key=lambda x: abs(x[1]), reverse=True)[:10]

    return AnalysisResult(
        prompt=prompt,
        verdict=verdict,
        confidence=confidence,
        reasoning=generated_text[:500],
        policy_attention_ratio=policy_ratio,
        content_attention_ratio=content_ratio,
        top_tokens=top_tokens,
        top_content_tokens=top_content_tokens,
        all_token_scores=all_token_scores,
        content_boundary_idx=content_boundary_idx,
        generated_text=generated_text,
    )
