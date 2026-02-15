#!/usr/bin/env python3
"""
AI Agent for evaluating persona video preferences using multiple LLM providers
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai

# Load environment variables
load_dotenv()


def generate_persona_display_name(persona: dict) -> str:
    """
    Generate a meaningful display name for a persona from demographics

    Args:
        persona: Persona dictionary with demographics

    Returns:
        Display name like "Male 25-34, Software Developer (San Francisco)"
    """
    # Check if persona has old-style 'name' field (for backwards compatibility)
    if 'name' in persona and persona['name']:
        return persona['name']

    # Generate name from demographics
    demo = persona.get('demographics', {})
    parts = []

    # Gender and age
    gender = demo.get('gender', [''])[0] if demo.get('gender') else ''
    age = demo.get('age', [''])[0] if demo.get('age') else ''
    if gender or age:
        parts.append(f"{gender} {age}".strip())

    # Career
    career = demo.get('careers', [''])[0] if demo.get('careers') else ''
    if career:
        # Pluralize career for generic description
        career_plural = career if career.endswith('s') else f"{career}s"
        parts.append(career_plural)

    # Location
    location = demo.get('locations', [''])[0] if demo.get('locations') else ''
    if location:
        parts.append(f"({location})")

    # If we have parts, join them
    if parts:
        return ", ".join(parts[:2]) + (" " + parts[2] if len(parts) > 2 else "")

    # Fallback: use first 60 chars of description
    description = persona.get('description', '')
    if description:
        return description[:60].strip() + ("..." if len(description) > 60 else "")

    # Ultimate fallback
    return f"Persona {persona.get('id', 'Unknown')}"

# Initialize API clients
openai.api_key = os.getenv('OPENAI_API_KEY')
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Available models (Updated January 2025)
MODELS = {
    'openai': {
        'gpt-5': 'gpt-5',
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini',
        # 'gpt-4-turbo': 'gpt-4-turbo',
        # 'gpt-4': 'gpt-4',
        'o1': 'o1',
        # 'o1-mini': 'o1-mini',
        # 'o1-preview': 'o1-preview',
        # 'o3-mini': 'o3-mini'
    },
    'anthropic': {
        'claude-sonnet-4.5': 'claude-sonnet-4-5-20250929',
        'claude-haiku-4.5': 'claude-haiku-4-5-20251001',
        'claude-opus-4.1': 'claude-opus-4-1-20250805',
        'claude-3.7-sonnet': 'claude-3-7-sonnet-20250219',
        # 'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
        # 'claude-3.5-haiku': 'claude-3-5-haiku-20241022',
        # 'claude-3-opus': 'claude-3-opus-20240229',
        # 'claude-3-sonnet': 'claude-3-sonnet-20240229',
        # 'claude-3-haiku': 'claude-3-haiku-20240307'
    },
    'google': {
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
        'gemini-2.0-flash': 'gemini-2.0-flash',
        'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite'
    }
}

def load_video_analyses():
    """Load all video analysis files"""
    analysis_dir = Path(__file__).parent / 'data'
    videos = []

    for i in range(1, 5):
        file_path = analysis_dir / f'video{i}.json'
        with open(file_path, 'r') as f:
            analysis = json.load(f)
            analysis['video_id'] = i
            videos.append(analysis)

    return videos

def load_personas():
    """Load persona data"""
    personas_file = Path(__file__).parent.parent / 'frontend' / 'public' / 'data' / 'customers' / 'personas.json'
    with open(personas_file, 'r') as f:
        return json.load(f)

def create_evaluation_prompt(persona, videos):
    """Create prompt for AI evaluation - handles any persona/video data format"""

    # Convert persona to readable format - just dump all available data
    persona_json = json.dumps(persona, indent=2, default=str)

    prompt = f"""You are analyzing video marketing preferences for a specific customer persona.

PERSONA DATA:
{persona_json}

VIDEOS TO ANALYZE (use the video number in your response):
"""

    # Assign simple numbers to videos and dump all available data
    for idx, video in enumerate(videos, 1):
        video_json = json.dumps(video, indent=2, default=str)
        prompt += f"""
========================================
VIDEO {idx}:
========================================
{video_json}

"""

    video_count = len(videos)
    video_numbers = ", ".join([str(i) for i in range(1, video_count + 1)])

    prompt += f"""
TASK:
Analyze the persona data (which may include demographics, behavior patterns, preferences, etc.) and the video data (which may include titles, analysis, metadata, etc.) to determine which video this persona would most likely prefer and engage with.

IMPORTANT: Use ONLY the video numbers (1, 2, 3, etc.) in your response, NOT the MongoDB IDs or any other identifiers.

Provide your analysis in the following JSON format:

{{
    "most_preferred_video": "1",
    "preference_ranking": ["1", "2", "3", "4"],
    "confidence_score": 85,
    "video_opinions": {{
        "1": "<opinion on video 1>",
        "2": "<opinion on video 2>",
        "3": "<opinion on video 3>",
        "4": "<opinion on video 4>"
    }},
    "reasoning": "<detailed explanation of why this persona would prefer certain videos>",
    "semantic_analysis": "<analysis of the semantic alignment between persona characteristics and video content>"
}}

Remember: Use video numbers ({video_numbers}) as strings in your response.

Consider all available data about the persona and videos to make your evaluation. Focus on:
1. Any demographic or behavioral information present in the persona data
2. Video content, style, messaging, and visual elements
3. Alignment between persona characteristics and video attributes
4. Likely engagement and preference patterns based on the available data
"""

    return prompt

def evaluate_with_openai(prompt, model_name='gpt-4o'):
    """Evaluate using OpenAI models"""
    try:
        # Some models (gpt-5, o1 series) don't support custom temperature
        models_without_temp = ['gpt-5', 'o1', 'o1-mini', 'o1-preview', 'o3-mini']

        params = {
            "model": MODELS['openai'][model_name],
            "messages": [
                {"role": "system", "content": "You are an expert marketing analyst specializing in customer persona analysis and video marketing effectiveness."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        # Only add temperature for models that support it
        if model_name not in models_without_temp:
            params["temperature"] = 0.7

        response = openai.chat.completions.create(**params)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error with OpenAI {model_name}: {e}")
        return None

def evaluate_with_anthropic(prompt, model_name='claude-3.5-sonnet'):
    """Evaluate using Anthropic Claude models"""
    try:
        response = anthropic_client.messages.create(
            model=MODELS['anthropic'][model_name],
            max_tokens=2048,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        # Extract JSON from response
        content = response.content[0].text
        # Try to find JSON in the response
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
        return None
    except Exception as e:
        print(f"Error with Anthropic {model_name}: {e}")
        return None

def evaluate_with_google(prompt, model_name='gemini-2.0-flash'):
    """Evaluate using Google Gemini models"""
    try:
        model = genai.GenerativeModel(MODELS['google'][model_name])
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error with Google {model_name}: {e}")
        return None

def evaluate_persona(persona, videos, provider='google', model_name=None):
    """
    Evaluate a persona's video preferences using specified AI model

    Args:
        persona: Persona object
        videos: List of video analysis objects
        provider: 'openai', 'anthropic', or 'google'
        model_name: Specific model name (optional, uses default if not specified)
    """
    prompt = create_evaluation_prompt(persona, videos)

    if provider == 'openai':
        model_name = model_name or 'gpt-4o'
        return evaluate_with_openai(prompt, model_name)
    elif provider == 'anthropic':
        model_name = model_name or 'claude-sonnet-4.5'
        return evaluate_with_anthropic(prompt, model_name)
    elif provider == 'google':
        model_name = model_name or 'gemini-2.5-pro'
        return evaluate_with_google(prompt, model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def run_evaluation(persona_id=None, provider='google', model_name=None, save_results=True):
    """
    Run evaluation for one or all personas

    Args:
        persona_id: Specific persona ID to evaluate (None for all)
        provider: AI provider to use
        model_name: Specific model name
        save_results: Whether to save results to file
    """
    print(f"Loading data...")
    videos = load_video_analyses()
    personas = load_personas()

    print(f"Loaded {len(videos)} videos and {len(personas)} personas")
    print(f"Using provider: {provider}, model: {model_name or 'default'}")
    print()

    results = []

    # Filter to specific persona if requested
    if persona_id:
        personas = [p for p in personas if p['id'] == persona_id]
        if not personas:
            print(f"Persona {persona_id} not found!")
            return

    for i, persona in enumerate(personas):
        persona_display_name = generate_persona_display_name(persona)
        print(f"Evaluating Persona {persona['id']}/{len(personas)}: {persona_display_name}...")

        evaluation = evaluate_persona(persona, videos, provider, model_name)

        if evaluation:
            result = {
                'persona_id': persona['id'],
                'persona_name': persona_display_name,
                'provider': provider,
                'model': model_name or f"{provider}_default",
                'evaluation': evaluation
            }
            results.append(result)

            print(f"  ✓ Most preferred: Video {evaluation.get('most_preferred_video')}")
            print(f"  ✓ Confidence: {evaluation.get('confidence_score')}%")
            print()
        else:
            print(f"  ✗ Evaluation failed")
            print()

    # Save results
    if save_results and results:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)

        model_suffix = model_name or f"{provider}_default"
        output_file = output_dir / f'persona_evaluations_{model_suffix}.json'

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")
        print(f"  Evaluated {len(results)} personas")

    return results

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate persona video preferences using AI')
    parser.add_argument('--persona-id', type=int, help='Specific persona ID to evaluate')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'google'], default='google',
                       help='AI provider to use')
    parser.add_argument('--model', help='Specific model name')
    parser.add_argument('--list-models', action='store_true', help='List available models')

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for provider, models in MODELS.items():
            print(f"\n{provider.upper()}:")
            for name in models.keys():
                print(f"  - {name}")
        return

    # Check API keys
    if args.provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in .env file")
        return
    elif args.provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        return
    elif args.provider == 'google' and not os.getenv('GOOGLE_API_KEY'):
        print("Error: GOOGLE_API_KEY not found in .env file")
        return

    run_evaluation(
        persona_id=args.persona_id,
        provider=args.provider,
        model_name=args.model
    )

if __name__ == '__main__':
    main()
