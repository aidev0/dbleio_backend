"""
Model Registry — Single source of truth for all AI models available in the platform.

Categories:
1. LLM (API-based) — For reasoning, planning, code review, QA
2. CLI (local dev tools) — For code generation, commits, PRs
3. Video Generation — For content production (direct API + Replicate)
4. Image Generation — For visual content production (direct API + Replicate)

All model identifiers are loaded from .env for easy configuration.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    id: str            # Internal identifier (used in DB and API)
    name: str          # Human-readable name
    provider: str      # Provider: openai, anthropic, google, runway, bytedance, kuaishou, luma, replicate
    model_string: str  # Actual API model string (for Replicate: owner/model-name)
    category: str      # "llm" | "cli" | "video" | "image"
    tier: str          # "standard" | "pro" | "enterprise"
    description: str
    enabled: bool = True
    env_key: Optional[str] = None  # .env key for API key
    platform: Optional[str] = None  # "direct" | "replicate" — how the model is accessed


# ─── LLM Models (API-based reasoning/coding) ───

LLM_MODELS: Dict[str, ModelConfig] = {
    "claude-sonnet-4-5": ModelConfig(
        id="claude-sonnet-4-5",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        model_string=os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
        category="llm",
        tier="standard",
        description="Fast, capable model for most development tasks.",
        env_key="ANTHROPIC_API_KEY",
        platform="direct",
    ),
    "claude-opus-4-6": ModelConfig(
        id="claude-opus-4-6",
        name="Claude Opus 4.6",
        provider="anthropic",
        model_string=os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-6"),
        category="llm",
        tier="pro",
        description="Most capable Anthropic model for complex reasoning and architecture.",
        env_key="ANTHROPIC_API_KEY",
        platform="direct",
    ),
    "gemini-3-pro": ModelConfig(
        id="gemini-3-pro",
        name="Gemini 3 Pro",
        provider="google",
        model_string=os.getenv("GEMINI_PRO_MODEL", "gemini-3-pro-preview"),
        category="llm",
        tier="pro",
        description="Google's most capable model for multimodal tasks.",
        env_key="GOOGLE_API_KEY",
        platform="direct",
    ),
    "gpt-5-2": ModelConfig(
        id="gpt-5-2",
        name="GPT-5.2",
        provider="openai",
        model_string=os.getenv("GPT_MODEL", "gpt-5.2"),
        category="llm",
        tier="pro",
        description="OpenAI's latest model for advanced reasoning.",
        env_key="OPENAI_API_KEY",
        platform="direct",
    ),
}

# ─── CLI Models (local code generation tools) ───

CLI_MODELS: Dict[str, ModelConfig] = {
    "claude-code": ModelConfig(
        id="claude-code",
        name="Claude Code",
        provider="anthropic",
        model_string="claude-code",
        category="cli",
        tier="pro",
        description="Anthropic's CLI for autonomous coding with Claude Opus.",
        env_key="ANTHROPIC_API_KEY",
        platform="direct",
    ),
    "gemini-cli": ModelConfig(
        id="gemini-cli",
        name="Gemini CLI",
        provider="google",
        model_string="gemini-cli",
        category="cli",
        tier="standard",
        description="Google's CLI for coding with Gemini models.",
        env_key="GOOGLE_API_KEY",
        platform="direct",
    ),
    "codex": ModelConfig(
        id="codex",
        name="Codex (OpenAI)",
        provider="openai",
        model_string="codex",
        category="cli",
        tier="standard",
        description="OpenAI's CLI for autonomous coding tasks.",
        env_key="OPENAI_API_KEY",
        platform="direct",
    ),
}

# ─── Video Generation Models ───

VIDEO_MODELS: Dict[str, ModelConfig] = {
    # --- Direct API models ---
    # OpenAI Sora
    "sora-2": ModelConfig(
        id="sora-2",
        name="Sora 2",
        provider="openai",
        model_string="sora-2",
        category="video",
        tier="standard",
        description="OpenAI Sora standard (720p max).",
        env_key="OPENAI_API_KEY",
        platform="direct",
    ),
    "sora-2-pro": ModelConfig(
        id="sora-2-pro",
        name="Sora 2 Pro",
        provider="openai",
        model_string="sora-2-pro",
        category="video",
        tier="pro",
        description="OpenAI Sora Pro (up to 1080p, higher quality).",
        env_key="OPENAI_API_KEY",
        platform="direct",
    ),
    # Google Veo
    "veo-3-1": ModelConfig(
        id="veo-3-1",
        name="Veo 3.1",
        provider="google",
        model_string="veo-3.1-generate-preview",
        category="video",
        tier="pro",
        description="Google Veo via Gemini API / Vertex AI.",
        env_key="GOOGLE_API_KEY",
        platform="direct",
    ),
    # Runway
    "gen-4-5": ModelConfig(
        id="gen-4-5",
        name="Runway Gen-4.5",
        provider="runway",
        model_string="Gen-4.5",
        category="video",
        tier="pro",
        description="Runway Gen-4.5 video generation.",
        env_key="RUNWAY_API_KEY",
        platform="direct",
    ),
    # ByteDance Seedance
    "seedance-1-0": ModelConfig(
        id="seedance-1-0",
        name="Seedance 1.0",
        provider="bytedance",
        model_string="seedance-1.0",
        category="video",
        tier="standard",
        description="ByteDance Seedance (Lite and Pro modes).",
        env_key="SEEDANCE_API_KEY",
        platform="direct",
    ),
    # Kling (Kuaishou)
    "kling-2-1": ModelConfig(
        id="kling-2-1",
        name="Kling 2.1",
        provider="kuaishou",
        model_string="kling-2.1",
        category="video",
        tier="standard",
        description="Kuaishou Kling 2.1 video generation.",
        env_key="KLING_API_KEY",
        platform="direct",
    ),
    "kling-o1": ModelConfig(
        id="kling-o1",
        name="Kling O1",
        provider="kuaishou",
        model_string="kling-o1",
        category="video",
        tier="pro",
        description="Kuaishou Kling O1 unified multimodal model.",
        env_key="KLING_API_KEY",
        platform="direct",
    ),
    # Luma Labs
    "ray-2": ModelConfig(
        id="ray-2",
        name="Ray 2",
        provider="luma",
        model_string="ray-2",
        category="video",
        tier="standard",
        description="Luma Labs Ray 2 video generation.",
        env_key="LUMA_API_KEY",
        platform="direct",
    ),
    "ray-3": ModelConfig(
        id="ray-3",
        name="Ray 3",
        provider="luma",
        model_string="ray-3",
        category="video",
        tier="pro",
        description="Luma Labs Ray 3 video generation.",
        env_key="LUMA_API_KEY",
        platform="direct",
    ),

    # --- Replicate-hosted Video Models ---
    "grok-magine-video": ModelConfig(
        id="grok-magine-video",
        name="Grok Magine Video",
        provider="replicate",
        model_string="grok-magine-video",
        category="video",
        tier="pro",
        description="Grok Magine video generation via Replicate.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Google Veo (via Replicate)
    "rep-veo-3-1": ModelConfig(
        id="rep-veo-3-1",
        name="Veo 3.1 (Replicate)",
        provider="replicate",
        model_string="google/veo-3.1",
        category="video",
        tier="pro",
        description="Google Veo 3.1 — higher-fidelity video, context-aware audio.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-veo-3-1-fast": ModelConfig(
        id="rep-veo-3-1-fast",
        name="Veo 3.1 Fast (Replicate)",
        provider="replicate",
        model_string="google/veo-3.1-fast",
        category="video",
        tier="standard",
        description="Faster Veo 3.1 with context-aware audio and last frame support.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-veo-3": ModelConfig(
        id="rep-veo-3",
        name="Veo 3 (Replicate)",
        provider="replicate",
        model_string="google/veo-3",
        category="video",
        tier="pro",
        description="Google Veo 3 text-to-video with audio.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-veo-3-fast": ModelConfig(
        id="rep-veo-3-fast",
        name="Veo 3 Fast (Replicate)",
        provider="replicate",
        model_string="google/veo-3-fast",
        category="video",
        tier="standard",
        description="Faster/cheaper Veo 3 with audio.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-veo-2": ModelConfig(
        id="rep-veo-2",
        name="Veo 2 (Replicate)",
        provider="replicate",
        model_string="google/veo-2",
        category="video",
        tier="standard",
        description="Google Veo 2 — faithfully follows instructions, simulates real-world physics.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Kling (Kuaishou via Replicate)
    "rep-kling-v2-6": ModelConfig(
        id="rep-kling-v2-6",
        name="Kling v2.6 (Replicate)",
        provider="replicate",
        model_string="kwaivgi/kling-v2.6",
        category="video",
        tier="pro",
        description="Top-tier image-to-video with cinematic visuals and native audio. Best for accurate physics.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-kling-v2-5-turbo-pro": ModelConfig(
        id="rep-kling-v2-5-turbo-pro",
        name="Kling v2.5 Turbo Pro (Replicate)",
        provider="replicate",
        model_string="kwaivgi/kling-v2.5-turbo-pro",
        category="video",
        tier="pro",
        description="Pro-level text-to-video and image-to-video with smooth motion and cinematic depth.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-kling-v2-1": ModelConfig(
        id="rep-kling-v2-1",
        name="Kling v2.1 (Replicate)",
        provider="replicate",
        model_string="kwaivgi/kling-v2.1",
        category="video",
        tier="standard",
        description="Kling v2.1 — 5s and 10s videos in 720p and 1080p.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-kling-v2-1-master": ModelConfig(
        id="rep-kling-v2-1-master",
        name="Kling v2.1 Master (Replicate)",
        provider="replicate",
        model_string="kwaivgi/kling-v2.1-master",
        category="video",
        tier="pro",
        description="Premium Kling v2.1 with superb dynamics and prompt adherence.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-kling-v2-0": ModelConfig(
        id="rep-kling-v2-0",
        name="Kling v2.0 (Replicate)",
        provider="replicate",
        model_string="kwaivgi/kling-v2.0",
        category="video",
        tier="standard",
        description="Generate 5s and 10s videos in 720p.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-kling-v1-6-pro": ModelConfig(
        id="rep-kling-v1-6-pro",
        name="Kling v1.6 Pro (Replicate)",
        provider="replicate",
        model_string="kwaivgi/kling-v1.6-pro",
        category="video",
        tier="pro",
        description="Generate 5s and 10s videos in 1080p.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-kling-v1-6-standard": ModelConfig(
        id="rep-kling-v1-6-standard",
        name="Kling v1.6 Standard (Replicate)",
        provider="replicate",
        model_string="kwaivgi/kling-v1.6-standard",
        category="video",
        tier="standard",
        description="720p at 30fps, 5s and 10s videos.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # OpenAI Sora (via Replicate)
    "rep-sora-2": ModelConfig(
        id="rep-sora-2",
        name="Sora 2 (Replicate)",
        provider="replicate",
        model_string="openai/sora-2",
        category="video",
        tier="standard",
        description="OpenAI Sora 2 — realistic/home video quality via Replicate.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-sora-2-pro": ModelConfig(
        id="rep-sora-2-pro",
        name="Sora 2 Pro (Replicate)",
        provider="replicate",
        model_string="openai/sora-2-pro",
        category="video",
        tier="pro",
        description="OpenAI Sora 2 Pro — most advanced synced-audio video generation.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # ByteDance Seedance (via Replicate)
    "rep-seedance-1-pro": ModelConfig(
        id="rep-seedance-1-pro",
        name="Seedance 1 Pro (Replicate)",
        provider="replicate",
        model_string="bytedance/seedance-1-pro",
        category="video",
        tier="pro",
        description="Pro version — 5s or 10s videos at 480p and 1080p.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-seedance-1-lite": ModelConfig(
        id="rep-seedance-1-lite",
        name="Seedance 1 Lite (Replicate)",
        provider="replicate",
        model_string="bytedance/seedance-1-lite",
        category="video",
        tier="standard",
        description="Lite — 5s or 10s videos at 480p and 720p.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-seedance-1-pro-fast": ModelConfig(
        id="rep-seedance-1-pro-fast",
        name="Seedance 1 Pro Fast (Replicate)",
        provider="replicate",
        model_string="bytedance/seedance-1-pro-fast",
        category="video",
        tier="standard",
        description="Faster and cheaper Seedance 1 Pro.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Wan Video (Alibaba, via Replicate)
    "rep-wan-2-5-t2v": ModelConfig(
        id="rep-wan-2-5-t2v",
        name="Wan 2.5 T2V (Replicate)",
        provider="replicate",
        model_string="wan-video/wan-2.5-t2v",
        category="video",
        tier="standard",
        description="Alibaba Wan 2.5 text-to-video.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-wan-2-5-i2v": ModelConfig(
        id="rep-wan-2-5-i2v",
        name="Wan 2.5 I2V (Replicate)",
        provider="replicate",
        model_string="wan-video/wan-2.5-i2v",
        category="video",
        tier="standard",
        description="Alibaba Wan 2.5 image-to-video with background audio.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-wan-2-2-t2v-fast": ModelConfig(
        id="rep-wan-2-2-t2v-fast",
        name="Wan 2.2 T2V Fast (Replicate)",
        provider="replicate",
        model_string="wan-video/wan-2.2-t2v-fast",
        category="video",
        tier="standard",
        description="Optimized fast Wan 2.2 A14B text-to-video.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-wan-2-2-i2v-fast": ModelConfig(
        id="rep-wan-2-2-i2v-fast",
        name="Wan 2.2 I2V Fast (Replicate)",
        provider="replicate",
        model_string="wan-video/wan-2.2-i2v-fast",
        category="video",
        tier="standard",
        description="Optimized fast Wan 2.2 A14B image-to-video.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Luma Labs (via Replicate)
    "rep-ray-2-720p": ModelConfig(
        id="rep-ray-2-720p",
        name="Ray 2 720p (Replicate)",
        provider="replicate",
        model_string="luma/ray-2-720p",
        category="video",
        tier="standard",
        description="Luma Ray 2 at 720p, 5s and 9s videos.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-ray-flash-2-720p": ModelConfig(
        id="rep-ray-flash-2-720p",
        name="Ray Flash 2 720p (Replicate)",
        provider="replicate",
        model_string="luma/ray-flash-2-720p",
        category="video",
        tier="standard",
        description="Faster and cheaper than Ray 2.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # PixVerse
    "rep-pixverse-v5": ModelConfig(
        id="rep-pixverse-v5",
        name="PixVerse v5 (Replicate)",
        provider="replicate",
        model_string="pixverse/pixverse-v5",
        category="video",
        tier="pro",
        description="5s-8s videos with enhanced character movement. Optimized for anime.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-pixverse-v4-5": ModelConfig(
        id="rep-pixverse-v4-5",
        name="PixVerse v4.5 (Replicate)",
        provider="replicate",
        model_string="pixverse/pixverse-v4.5",
        category="video",
        tier="standard",
        description="5s or 8s videos with enhanced motion and prompt coherence.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # MiniMax Hailuo
    "rep-hailuo-2-3": ModelConfig(
        id="rep-hailuo-2-3",
        name="Hailuo 2.3 (Replicate)",
        provider="replicate",
        model_string="minimax/hailuo-2.3",
        category="video",
        tier="pro",
        description="High-fidelity video — realistic human motion, cinematic VFX.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "rep-hailuo-02": ModelConfig(
        id="rep-hailuo-02",
        name="Hailuo 02 (Replicate)",
        provider="replicate",
        model_string="minimax/hailuo-02",
        category="video",
        tier="standard",
        description="T2V and I2V, 6s or 10s at 768p (standard) or 1080p (pro). Excels at real world physics.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # VEED Fabric
    "rep-fabric-1-0": ModelConfig(
        id="rep-fabric-1-0",
        name="VEED Fabric 1.0 (Replicate)",
        provider="replicate",
        model_string="veed/fabric-1.0",
        category="video",
        tier="standard",
        description="Image-to-video API — turns any image into a talking video.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Tencent Hunyuan
    "rep-hunyuan-video": ModelConfig(
        id="rep-hunyuan-video",
        name="Hunyuan Video (Replicate)",
        provider="replicate",
        model_string="tencent/hunyuan-video",
        category="video",
        tier="standard",
        description="Tencent's text-to-video model — high-quality with realistic motion.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Leonardo AI
    "rep-motion-2-0": ModelConfig(
        id="rep-motion-2-0",
        name="Motion 2.0 (Replicate)",
        provider="replicate",
        model_string="leonardoai/motion-2.0",
        category="video",
        tier="standard",
        description="Create 5s 480p videos from text prompt.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
}

# ─── Image Generation Models (Replicate) ───

IMAGE_MODELS: Dict[str, ModelConfig] = {
    # Google
    "nano-banana-pro": ModelConfig(
        id="nano-banana-pro",
        name="Nano Banana Pro (Google)",
        provider="replicate",
        model_string="google/nano-banana-pro",
        category="image",
        tier="pro",
        description="Google's SOTA image generation and editing model. Best overall + best text rendering.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "nano-banana": ModelConfig(
        id="nano-banana",
        name="Nano Banana (Google)",
        provider="replicate",
        model_string="google/nano-banana",
        category="image",
        tier="standard",
        description="Google's latest image editing model in Gemini 2.5.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "imagen-4": ModelConfig(
        id="imagen-4",
        name="Imagen 4 (Google)",
        provider="replicate",
        model_string="google/imagen-4",
        category="image",
        tier="pro",
        description="Google's Imagen 4 flagship image model.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "imagen-4-fast": ModelConfig(
        id="imagen-4-fast",
        name="Imagen 4 Fast (Google)",
        provider="replicate",
        model_string="google/imagen-4-fast",
        category="image",
        tier="standard",
        description="Fast version of Imagen 4 — speed and cost prioritized.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "imagen-4-ultra": ModelConfig(
        id="imagen-4-ultra",
        name="Imagen 4 Ultra (Google)",
        provider="replicate",
        model_string="google/imagen-4-ultra",
        category="image",
        tier="enterprise",
        description="Ultra Imagen 4 — quality matters more than speed and cost.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Pruna AI
    "p-image": ModelConfig(
        id="p-image",
        name="P-Image (Pruna AI)",
        provider="replicate",
        model_string="prunaai/p-image",
        category="image",
        tier="standard",
        description="Sub 1-second text-to-image for production use cases.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "z-image-turbo": ModelConfig(
        id="z-image-turbo",
        name="Z-Image Turbo (Pruna AI)",
        provider="replicate",
        model_string="prunaai/z-image-turbo",
        category="image",
        tier="standard",
        description="Super fast 6B parameter text-to-image model.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # ByteDance Seedream
    "seedream-4-5": ModelConfig(
        id="seedream-4-5",
        name="Seedream 4.5 (ByteDance)",
        provider="replicate",
        model_string="bytedance/seedream-4.5",
        category="image",
        tier="pro",
        description="Upgraded with stronger spatial understanding and world knowledge.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "seedream-4": ModelConfig(
        id="seedream-4",
        name="Seedream 4 (ByteDance)",
        provider="replicate",
        model_string="bytedance/seedream-4",
        category="image",
        tier="standard",
        description="Unified text-to-image + editing at up to 4K resolution.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Qwen
    "qwen-image": ModelConfig(
        id="qwen-image",
        name="Qwen Image",
        provider="replicate",
        model_string="qwen/qwen-image",
        category="image",
        tier="pro",
        description="Image generation with advances in complex text rendering.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Black Forest Labs FLUX
    "flux-2-max": ModelConfig(
        id="flux-2-max",
        name="FLUX 2 Max",
        provider="replicate",
        model_string="black-forest-labs/flux-2-max",
        category="image",
        tier="enterprise",
        description="Highest fidelity image model from Black Forest Labs.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "flux-pro": ModelConfig(
        id="flux-pro",
        name="FLUX Pro",
        provider="replicate",
        model_string="black-forest-labs/flux-pro",
        category="image",
        tier="pro",
        description="SOTA prompt following, visual quality, image detail and output diversity.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "flux-schnell": ModelConfig(
        id="flux-schnell",
        name="FLUX Schnell",
        provider="replicate",
        model_string="black-forest-labs/flux-schnell",
        category="image",
        tier="standard",
        description="Fastest image generation model for local dev and personal use.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "flux-kontext-max": ModelConfig(
        id="flux-kontext-max",
        name="FLUX Kontext Max",
        provider="replicate",
        model_string="black-forest-labs/flux-kontext-max",
        category="image",
        tier="enterprise",
        description="Premium text-based image editing with improved typography.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "flux-kontext-pro": ModelConfig(
        id="flux-kontext-pro",
        name="FLUX Kontext Pro",
        provider="replicate",
        model_string="black-forest-labs/flux-kontext-pro",
        category="image",
        tier="pro",
        description="SOTA text-based image editing through natural language.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Ideogram
    "ideogram-v3-turbo": ModelConfig(
        id="ideogram-v3-turbo",
        name="Ideogram v3 Turbo",
        provider="replicate",
        model_string="ideogram-ai/ideogram-v3-turbo",
        category="image",
        tier="standard",
        description="Fastest and cheapest Ideogram v3 with stunning realism.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "ideogram-v3-quality": ModelConfig(
        id="ideogram-v3-quality",
        name="Ideogram v3 Quality",
        provider="replicate",
        model_string="ideogram-ai/ideogram-v3-quality",
        category="image",
        tier="pro",
        description="Highest quality Ideogram v3 — stunning realism and creative designs.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Recraft
    "recraft-v3": ModelConfig(
        id="recraft-v3",
        name="Recraft V3",
        provider="replicate",
        model_string="recraft-ai/recraft-v3",
        category="image",
        tier="pro",
        description="SOTA text-to-image with long text and wide style support.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "recraft-v3-svg": ModelConfig(
        id="recraft-v3-svg",
        name="Recraft V3 SVG",
        provider="replicate",
        model_string="recraft-ai/recraft-v3-svg",
        category="image",
        tier="pro",
        description="Generate high-quality SVG images including logotypes and icons.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Stability AI
    "sdxl": ModelConfig(
        id="sdxl",
        name="SDXL (Stability AI)",
        provider="replicate",
        model_string="stability-ai/sdxl",
        category="image",
        tier="standard",
        description="Text-to-image generative AI model that creates beautiful images.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    "sd-3-5-large": ModelConfig(
        id="sd-3-5-large",
        name="SD 3.5 Large (Stability AI)",
        provider="replicate",
        model_string="stability-ai/stable-diffusion-3.5-large",
        category="image",
        tier="pro",
        description="High-resolution images with fine details and diverse outputs.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # MiniMax
    "image-01": ModelConfig(
        id="image-01",
        name="Image-01 (MiniMax)",
        provider="replicate",
        model_string="minimax/image-01",
        category="image",
        tier="standard",
        description="MiniMax's first image model with character reference support.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Luma
    "photon": ModelConfig(
        id="photon",
        name="Photon (Luma)",
        provider="replicate",
        model_string="luma/photon",
        category="image",
        tier="pro",
        description="High-quality image generation for creative professional workflows.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
    # Bria
    "bria-fibo": ModelConfig(
        id="bria-fibo",
        name="FIBO (Bria)",
        provider="replicate",
        model_string="bria/fibo",
        category="image",
        tier="enterprise",
        description="SOTA open source model trained on licensed data for enterprise workflows.",
        env_key="REPLICATE_API_TOKEN",
        platform="replicate",
    ),
}

# ─── Combined Registry ───

ALL_MODELS: Dict[str, ModelConfig] = {**LLM_MODELS, **CLI_MODELS, **VIDEO_MODELS, **IMAGE_MODELS}


def get_model(model_id: str) -> Optional[ModelConfig]:
    return ALL_MODELS.get(model_id)


def get_models_by_category(category: str) -> List[ModelConfig]:
    return [m for m in ALL_MODELS.values() if m.category == category and m.enabled]


def get_available_llms() -> List[dict]:
    return [{"id": m.id, "name": m.name, "provider": m.provider, "tier": m.tier, "description": m.description}
            for m in get_models_by_category("llm")]


def get_available_clis() -> List[dict]:
    return [{"id": m.id, "name": m.name, "provider": m.provider, "tier": m.tier, "description": m.description}
            for m in get_models_by_category("cli")]


def get_available_video_models() -> List[dict]:
    return [{"id": m.id, "name": m.name, "provider": m.provider, "tier": m.tier,
             "description": m.description, "platform": m.platform}
            for m in get_models_by_category("video")]


def get_available_image_models() -> List[dict]:
    return [{"id": m.id, "name": m.name, "provider": m.provider, "tier": m.tier,
             "description": m.description, "platform": m.platform}
            for m in get_models_by_category("image")]


def get_replicate_models() -> List[dict]:
    """Get all models that run via Replicate."""
    return [{"id": m.id, "name": m.name, "category": m.category, "model_string": m.model_string,
             "tier": m.tier, "description": m.description}
            for m in ALL_MODELS.values() if m.platform == "replicate" and m.enabled]


def is_model_available(model_id: str) -> bool:
    """Check if a model's API key is configured."""
    model = get_model(model_id)
    if not model:
        return False
    if model.env_key:
        return bool(os.getenv(model.env_key))
    return True


# ─── Default model selections per stage type ───

DEFAULT_DEV_LLM = os.getenv("DEFAULT_DEV_LLM", "claude-sonnet-4-5")
DEFAULT_DEV_CLI = os.getenv("DEFAULT_DEV_CLI", "claude-code")
DEFAULT_VIDEO_MODEL = os.getenv("DEFAULT_VIDEO_MODEL", "sora-2")
DEFAULT_IMAGE_MODEL = os.getenv("DEFAULT_IMAGE_MODEL", "nano-banana-pro")
