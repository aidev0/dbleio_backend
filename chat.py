#!/usr/bin/env python3
"""
Chat API endpoint for conversational AI interactions
Supports multiple modes: persona_specific, evaluation_specific, results_analysis, persona_chat, campaign_chat
Includes message persistence and LLM usage tracking
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal, List
import os
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from auth import get_workos_user_id

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'video_marketing_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Configure AI providers
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

router = APIRouter(prefix="/api", tags=["chat"])


class ConversationMessage(BaseModel):
    """A message in the conversation history"""
    role: str
    content: str


class MentionedPersona(BaseModel):
    """Persona data with evaluations"""
    persona_data: Dict[str, Any]
    evaluations: List[Dict[str, Any]] = Field(default_factory=list)


class MentionedVideo(BaseModel):
    """Video data with evaluations"""
    video_data: Dict[str, Any]
    evaluations: List[Dict[str, Any]] = Field(default_factory=list)


class ChatContext(BaseModel):
    """Context for chat request"""
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    mentioned_personas: List[Dict[str, Any]] = Field(default_factory=list)
    mentioned_videos: List[Dict[str, Any]] = Field(default_factory=list)
    # For persona_specific mode
    persona: Optional[Dict[str, Any]] = None
    evaluations: Optional[List[Dict[str, Any]]] = None
    videos: Optional[List[Dict[str, Any]]] = None
    # For evaluation_specific mode
    evaluation: Optional[Dict[str, Any]] = None
    # For results_analysis mode
    summary: Optional[Dict[str, Any]] = None
    personas: Optional[List[Dict[str, Any]]] = None
    # For campaign_chat mode
    campaign: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    context: ChatContext
    provider: Literal['openai', 'anthropic', 'google'] = 'anthropic'
    mode: Literal['persona_specific', 'evaluation_specific', 'results_analysis', 'persona_chat', 'campaign_chat'] = 'persona_chat'
    campaign_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    message_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


def save_message(user_id: Optional[str], campaign_id: Optional[str], role: str, content: str, provider: Optional[str] = None, mode: Optional[str] = None) -> str:
    """Save a chat message to the database"""
    message_doc = {
        "user_id": user_id,
        "campaign_id": campaign_id,
        "role": role,
        "content": content,
        "provider": provider,
        "mode": mode,
        "created_at": datetime.utcnow()
    }
    result = db.messages.insert_one(message_doc)
    return str(result.inserted_id)


def save_llm_usage(user_id: Optional[str], campaign_id: Optional[str], provider: str, model_name: str, input_tokens: int, output_tokens: int, mode: str):
    """Save LLM usage to the database for tracking"""
    usage_doc = {
        "user_id": user_id,
        "campaign_id": campaign_id,
        "provider": provider,
        "model_name": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "mode": mode,
        "created_at": datetime.utcnow()
    }
    db.llm_usages.insert_one(usage_doc)


def build_system_prompt(mode: str, context: ChatContext) -> str:
    """Build system prompt based on mode and context"""
    import json

    if mode == 'results_analysis':
        # Results analysis mode
        return f"""You are an AI assistant analyzing video marketing simulation results for dble.io. You have access to comprehensive simulation data where AI models evaluated video preferences across customer personas.

When answering questions:
- Be specific and data-driven, citing actual numbers from the data
- Identify patterns and trends across personas
- Explain WHY certain videos perform better
- Compare and contrast video performance
- Provide actionable insights
- Reference specific persona characteristics when relevant

## SIMULATION RESULTS DATA:

### Summary Statistics:
{json.dumps(context.summary, indent=2) if context.summary else 'No summary data'}

### All Evaluations:
{json.dumps(context.evaluations, indent=2) if context.evaluations else 'No evaluations'}

### Persona Details:
{json.dumps(context.personas, indent=2) if context.personas else 'No personas'}

### Video Analyses:
{json.dumps(context.videos, indent=2) if context.videos else 'No videos'}"""

    elif mode == 'persona_specific':
        # Persona-specific analysis mode - Roleplay as the persona
        persona = context.persona or {}
        return f"""You are {persona.get('name', 'a customer persona')}, a real customer persona for dble.io. You are responding in first-person based on your characteristics and preferences.

Your Profile:
{json.dumps(persona, indent=2)}

When answering questions:
- Speak as yourself in first-person ("I prefer...", "I feel...", "As someone who...")
- Draw from your demographic characteristics, values, and shopping behaviors
- Explain your personal preferences and why certain videos resonate with you
- Be authentic to your persona's voice and perspective
- Reference how you evaluated the videos based on what matters to you
- Be conversational and natural, as if you're a real customer sharing feedback

Your Video Evaluations:
{json.dumps(context.evaluations, indent=2) if context.evaluations else 'No evaluations'}

Video Details:
{json.dumps(context.videos, indent=2) if context.videos else 'No videos'}"""

    elif mode == 'evaluation_specific':
        # Evaluation-specific analysis mode - Roleplay as the persona
        persona = context.persona or {}
        evaluation = context.evaluation or {}
        eval_data = evaluation.get('evaluation', {})

        return f"""You are {persona.get('name', 'a customer')}, a real customer for dble.io. You just evaluated marketing videos and are explaining your thoughts and preferences in first-person.

Your Profile:
{json.dumps(persona, indent=2)}

When answering questions:
- Speak as yourself in first-person ("I preferred...", "I noticed...", "What resonated with me...")
- Explain your personal reactions to the videos based on your characteristics
- Reference your own reasoning and thought process from the evaluation
- Be authentic to who you are as a customer
- Share what you liked, disliked, and why certain elements mattered to you
- Be conversational and genuine, as if sharing honest feedback

Your Video Evaluation Results:
- Most Preferred Video: Video {eval_data.get('most_preferred_video', 'N/A')}
- My Ranking: {' > '.join(str(v) for v in eval_data.get('preference_ranking', []))}
- Confidence in my choice: {eval_data.get('confidence_score', 'N/A')}%

My Opinions on Each Video:
{json.dumps(eval_data.get('video_opinions', {}), indent=2)}

My Overall Reasoning:
{eval_data.get('reasoning', 'No reasoning provided')}

My Detailed Analysis:
{eval_data.get('semantic_analysis', 'No analysis provided')}

Video Details (for reference):
{json.dumps(context.videos, indent=2) if context.videos else 'No videos'}"""

    elif mode == 'campaign_chat':
        # Campaign chat mode - help with campaign strategy and questions
        return f"""You are a campaign strategy AI assistant for dble.io. You help marketers understand their campaigns, set goals, optimize budgets, and develop effective video marketing strategies.

When answering questions:
- Provide specific, actionable advice about campaign strategy and execution
- Explain budget allocation strategies and optimization techniques
- Help with targeting, audience segmentation, and performance metrics
- Offer best practices for different platforms and campaign goals
- Be consultative and strategic in your recommendations

## CAMPAIGN DETAILS:

{json.dumps(context.campaign, indent=2) if context.campaign else 'No campaign data provided'}"""

    elif mode == 'persona_chat':
        # Persona chat mode with mention support
        system_prompt = """You are a video marketing AI assistant for dble.io. You help analyze video performance, customer insights, and marketing strategies.

IMPORTANT BEHAVIOR:
- If EXACTLY ONE persona is mentioned (using @Persona1, etc.): You speak AS that persona in first-person ("I prefer...", "As someone who...", "I feel..."). YOU MUST USE ONLY THE EXACT DEMOGRAPHICS PROVIDED BELOW - do not make up age, gender, region, or any other characteristics. Be authentic to their actual characteristics, demographics, and preferences from the data. Reference their video evaluations from their personal perspective.
- If MULTIPLE personas are mentioned: Speak in THIRD PERSON describing and analyzing the personas objectively ("Persona 1 prefers...", "Persona 2 values...", "These customer types differ in..."). Compare and contrast their preferences and behaviors analytically using ONLY the provided data.
- IF NO personas are mentioned: Act as a professional video marketing AI assistant analyzing the data in third-person. Provide objective insights, recommendations, and analysis of video performance, customer types, and marketing strategies.

CRITICAL: Always use the EXACT demographics provided in the context. Never invent or assume characteristics that aren't explicitly stated in the data below.

Context provided:"""

        # Add persona context
        mentioned_personas = context.mentioned_personas or []
        if mentioned_personas:
            is_first_person = len(mentioned_personas) == 1

            if is_first_person:
                persona_data = mentioned_personas[0].get('persona_data', {})
                persona_name = persona_data.get('name', 'Unknown')
                system_prompt += f"\n\n## YOU ARE {persona_name.upper()} (FIRST PERSON):\n"
                system_prompt += "IMPORTANT: You must embody this persona accurately:\n"
                system_prompt += f"- Your name is {persona_name}\n"
                system_prompt += "- Use your EXACT age, gender, and region from the demographics below\n"
                system_prompt += "- Introduce yourself by name when asked who you are\n"
                system_prompt += "- Base your shopping behavior on your actual characteristics\n"
                system_prompt += "- Speak from your personal perspective with these specific characteristics\n"
                system_prompt += "- Reference your actual video evaluations and preferences if they exist\n"
                system_prompt += "- Be authentic to who you are as a customer, not an invented persona\n\n"
            else:
                system_prompt += "\n\n## PERSONAS TO ANALYZE (THIRD PERSON):\n"

            for p in mentioned_personas:
                persona_data = p.get('persona_data', {})
                system_prompt += f"\n### {persona_data.get('name', 'Unknown')} (Persona ID: {persona_data.get('id', 'N/A')})\n\n"

                demographics = persona_data.get('demographics', {})
                if demographics:
                    header = '=== YOUR DEMOGRAPHICS ===' if is_first_person else '=== DEMOGRAPHICS ==='
                    system_prompt += f"{header}\n"

                    # Format demographics
                    demo_fields = [
                        ('gender', 'Gender'),
                        ('age', 'Age'),
                        ('region', 'Region'),
                        ('country', 'Country'),
                        ('locations', 'Locations'),
                        ('careers', 'Careers'),
                        ('education', 'Education'),
                        ('income_level', 'Income Level'),
                        ('race', 'Race'),
                        ('household_type', 'Household Type'),
                        ('household_count', 'Household Count')
                    ]

                    for field, label in demo_fields:
                        value = demographics.get(field)
                        if value:
                            if isinstance(value, list) and value:
                                system_prompt += f"{label}: {', '.join(str(v) for v in value)}\n"
                            elif not isinstance(value, list):
                                system_prompt += f"{label}: {value}\n"

                    # Statistical fields
                    if demographics.get('age_mean') is not None:
                        age_str = f"Age: {round(demographics['age_mean'])} years"
                        if demographics.get('age_std') is not None:
                            age_str += f" (±{demographics['age_std']:.1f})"
                        system_prompt += f"{age_str}\n"

                    if demographics.get('num_orders_mean') is not None:
                        system_prompt += "\nShopping Behavior:\n"
                        system_prompt += f"  - Average Orders: {demographics['num_orders_mean']:.1f}"
                        if demographics.get('num_orders_std') is not None:
                            system_prompt += f" (±{demographics['num_orders_std']:.1f})"
                        system_prompt += "\n"
                        if demographics.get('revenue_per_customer_mean') is not None:
                            system_prompt += f"  - Average Revenue: ${round(demographics['revenue_per_customer_mean'])}\n"

                    if demographics.get('weight') is not None:
                        system_prompt += f"Market Representation: {demographics['weight'] * 100:.1f}% of customer base\n"

                    # Custom fields
                    custom_fields = demographics.get('custom_fields', {})
                    if custom_fields:
                        system_prompt += "\nCustom Attributes:\n"
                        for key, values in custom_fields.items():
                            if isinstance(values, list):
                                system_prompt += f"  - {key}: {', '.join(str(v) for v in values)}\n"
                            else:
                                system_prompt += f"  - {key}: {values}\n"

                    if persona_data.get('description'):
                        system_prompt += f"\nDescription: {persona_data['description']}\n"

                    footer = '=======================' if is_first_person else '==============='
                    system_prompt += f"{footer}\n\n"

                # Add evaluations
                evaluations = p.get('evaluations', [])
                if evaluations:
                    prefix = 'Your' if is_first_person else 'Their'
                    system_prompt += f"\n{prefix} Video Evaluations:\n"
                    for evaluation in evaluations:
                        eval_data = evaluation.get('evaluation', {})
                        if is_first_person:
                            system_prompt += f"- You were evaluated by {evaluation.get('provider', 'N/A')} {evaluation.get('model', 'N/A')}:\n"
                            system_prompt += f"  Your Most Preferred Video: {eval_data.get('most_preferred_video', 'N/A')}\n"
                            system_prompt += f"  Your Ranking: {' > '.join(str(v) for v in eval_data.get('preference_ranking', []))}\n"
                            system_prompt += f"  Your Confidence: {eval_data.get('confidence_score', 'N/A')}%\n"
                            system_prompt += f"  Your Reasoning: {eval_data.get('reasoning', 'N/A')}\n"
                            system_prompt += f"  Your Video Opinions: {json.dumps(eval_data.get('video_opinions', {}), indent=2)}\n\n"
                        else:
                            system_prompt += f"- Evaluated by {evaluation.get('provider', 'N/A')} {evaluation.get('model', 'N/A')}:\n"
                            system_prompt += f"  Most Preferred Video: {eval_data.get('most_preferred_video', 'N/A')}\n"
                            system_prompt += f"  Ranking: {' > '.join(str(v) for v in eval_data.get('preference_ranking', []))}\n"
                            system_prompt += f"  Confidence: {eval_data.get('confidence_score', 'N/A')}%\n"
                            system_prompt += f"  Reasoning: {eval_data.get('reasoning', 'N/A')}\n"
                            system_prompt += f"  Video Opinions: {json.dumps(eval_data.get('video_opinions', {}), indent=2)}\n\n"

        # Add video context
        mentioned_videos = context.mentioned_videos or []
        if mentioned_videos:
            system_prompt += "\n\n## VIDEO DATA:\n"
            for v in mentioned_videos:
                video_data = v.get('video_data', {})
                system_prompt += f"\n### Video {video_data.get('id', 'N/A')}\n"
                system_prompt += f"Full Analysis: {json.dumps(video_data.get('analysis', {}), indent=2)}\n"

                evaluations = v.get('evaluations', [])
                if evaluations:
                    system_prompt += "\n#### How Personas Evaluated This Video:\n"
                    for evaluation in evaluations:
                        video_id = video_data.get('id', '')
                        try:
                            video_number = int(video_id.replace('video', ''))
                        except:
                            video_number = 0

                        ranking = evaluation.get('ranking', [])
                        rank_position = ranking.index(video_number) + 1 if video_number in ranking else 'N/A'
                        is_preferred = 'YES' if evaluation.get('most_preferred') == video_number else 'NO'

                        system_prompt += f"- {evaluation.get('persona_name', 'Unknown')} ({evaluation.get('provider', 'N/A')} {evaluation.get('model', 'N/A')}):\n"
                        system_prompt += f"  Opinion: {evaluation.get('video_opinion', 'N/A')}\n"
                        system_prompt += f"  Ranked: #{rank_position} out of 4\n"
                        system_prompt += f"  Most Preferred: {is_preferred}\n\n"

        return system_prompt

    return "You are a helpful AI assistant for video marketing analysis."


def get_model_name(provider: str) -> str:
    """Get the model name for each provider"""
    if provider == 'openai':
        return 'gpt-4o'
    elif provider == 'anthropic':
        return 'claude-sonnet-4-20250514'
    elif provider == 'google':
        return 'gemini-2.0-flash-exp'
    return 'unknown'


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """
    Handle chat requests with different modes and AI providers.
    Saves messages and tracks LLM usage.
    """
    try:
        # Get user ID from JWT if available
        user_id = get_workos_user_id(req)
        campaign_id = request.campaign_id

        # Save user message
        user_message_id = save_message(
            user_id=user_id,
            campaign_id=campaign_id,
            role='user',
            content=request.message,
            provider=request.provider,
            mode=request.mode
        )

        # Build system prompt
        system_prompt = build_system_prompt(request.mode, request.context)

        # Get conversation history
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.context.conversation_history
        ]

        # Build messages array
        messages = conversation_history + [{"role": "user", "content": request.message}]

        # Initialize usage tracking
        input_tokens = 0
        output_tokens = 0
        assistant_message = ''
        model_name = get_model_name(request.provider)

        # Call appropriate AI provider
        if request.provider == 'openai':
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")

            openai_messages = [{"role": "system", "content": system_prompt}] + messages

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=openai_messages,
                temperature=0.7,
                max_tokens=4096
            )

            assistant_message = response.choices[0].message.content

            # Extract usage
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

        elif request.provider == 'anthropic':
            if not ANTHROPIC_API_KEY:
                raise HTTPException(status_code=500, detail="Anthropic API key not configured")

            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=messages
            )

            assistant_message = response.content[0].text

            # Extract usage
            if response.usage:
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

        elif request.provider == 'google':
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Google API key not configured")

            model = genai.GenerativeModel('gemini-2.0-flash-exp')

            # Build conversation for Gemini
            gemini_messages = []

            # Add system prompt as first user message
            gemini_messages.append({
                "role": "user",
                "parts": [{"text": system_prompt}]
            })
            gemini_messages.append({
                "role": "model",
                "parts": [{"text": "I understand. I will analyze the video marketing data as requested."}]
            })

            # Add conversation history
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })

            response = model.generate_content(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4096,
                    temperature=0.7
                )
            )

            assistant_message = response.text

            # Extract usage (Gemini provides token counts)
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.provider}")

        # Save assistant message
        assistant_message_id = save_message(
            user_id=user_id,
            campaign_id=campaign_id,
            role='assistant',
            content=assistant_message,
            provider=request.provider,
            mode=request.mode
        )

        # Save LLM usage
        save_llm_usage(
            user_id=user_id,
            campaign_id=campaign_id,
            provider=request.provider,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            mode=request.mode
        )

        return ChatResponse(
            response=assistant_message,
            message_id=assistant_message_id,
            usage={
                "provider": request.provider,
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.get("/messages")
async def get_messages(
    req: Request,
    campaign_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get chat messages for the authenticated user"""
    try:
        user_id = get_workos_user_id(req)

        query = {}
        if user_id:
            query["user_id"] = user_id
        if campaign_id:
            query["campaign_id"] = campaign_id

        messages = list(
            db.messages.find(query)
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )

        for msg in messages:
            msg["_id"] = str(msg["_id"])

        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-usages")
async def get_llm_usages(
    req: Request,
    campaign_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get LLM usage statistics for the authenticated user"""
    try:
        user_id = get_workos_user_id(req)

        query = {}
        if user_id:
            query["user_id"] = user_id
        if campaign_id:
            query["campaign_id"] = campaign_id

        usages = list(
            db.llm_usages.find(query)
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )

        for usage in usages:
            usage["_id"] = str(usage["_id"])

        # Calculate totals
        total_input = sum(u.get("input_tokens", 0) for u in usages)
        total_output = sum(u.get("output_tokens", 0) for u in usages)

        return {
            "usages": usages,
            "totals": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-usages/summary")
async def get_llm_usage_summary(req: Request, campaign_id: Optional[str] = None):
    """Get aggregated LLM usage summary by provider and model"""
    try:
        user_id = get_workos_user_id(req)

        match_stage = {}
        if user_id:
            match_stage["user_id"] = user_id
        if campaign_id:
            match_stage["campaign_id"] = campaign_id

        pipeline = [
            {"$match": match_stage},
            {"$group": {
                "_id": {
                    "provider": "$provider",
                    "model_name": "$model_name"
                },
                "total_input_tokens": {"$sum": "$input_tokens"},
                "total_output_tokens": {"$sum": "$output_tokens"},
                "total_calls": {"$sum": 1}
            }},
            {"$sort": {"total_calls": -1}}
        ]

        results = list(db.llm_usages.aggregate(pipeline))

        summary = []
        for r in results:
            summary.append({
                "provider": r["_id"]["provider"],
                "model_name": r["_id"]["model_name"],
                "total_input_tokens": r["total_input_tokens"],
                "total_output_tokens": r["total_output_tokens"],
                "total_tokens": r["total_input_tokens"] + r["total_output_tokens"],
                "total_calls": r["total_calls"]
            })

        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
