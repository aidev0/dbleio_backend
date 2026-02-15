#!/usr/bin/env python3
"""
AI-powered persona generation for marketing campaigns
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize API clients
openai.api_key = os.getenv('OPENAI_API_KEY')
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Available models
MODELS = {
    'openai': {
        'gpt-5': 'gpt-5',
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini',
        'o1': 'o1',
    },
    'anthropic': {
        'claude-sonnet-4.5': 'claude-sonnet-4-5-20250929',
        'claude-haiku-4.5': 'claude-haiku-4-5-20251001',
        'claude-opus-4.1': 'claude-opus-4-1-20250805',
        'claude-3.7-sonnet': 'claude-3-7-sonnet-20250219',
    },
    'google': {
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
        'gemini-2.0-flash': 'gemini-2.0-flash',
        'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite'
    }
}

def create_persona_generation_prompt(campaign: Optional[Dict[str, Any]], num_personas: int, selected_dimensions: Optional[List[str]] = None, distribution_description: Optional[str] = None) -> str:
    """Create prompt for AI persona generation"""
    if campaign:
        campaign_name = campaign.get('name', 'Unknown Campaign')
        campaign_desc = campaign.get('description', 'No description provided')
        product_type = campaign.get('product_type', 'product')
        target_market = campaign.get('target_market', 'general consumers')

        campaign_details = f"""CAMPAIGN DETAILS:
- Campaign Name: {campaign_name}
- Description: {campaign_desc}
- Product Type: {product_type}
- Target Market: {target_market}"""
    else:
        campaign_details = """CAMPAIGN DETAILS:
- General marketing personas for diverse audience segments
- No specific campaign context"""

    # Add distribution guidance if provided
    if distribution_description:
        campaign_details += f"\n\nDISTRIBUTION GUIDANCE:\n{distribution_description}"

    # Define available demographic fields and their guidelines
    demographic_fields = {
        'age': {
            'example': '["25-34"]',
            'guideline': 'Age: Each persona should have exactly ONE age range from: 18-24, 25-34, 35-44, 45-54, 55-64, 65+. Distribute varied ages across personas.'
        },
        'gender': {
            'example': '["Male"]',
            'guideline': 'Gender: Each persona should have exactly ONE gender - either Male or Female (not both). Distribute evenly across personas.'
        },
        'locations': {
            'example': '["New York", "Los Angeles"]',
            'guideline': 'Locations: Urban, suburban, rural areas across different regions'
        },
        'country': {
            'example': '["United States"]',
            'guideline': 'Country: Each persona should have exactly ONE country where they reside. Vary countries across personas based on campaign relevance.'
        },
        'region': {
            'example': '["Northeast"]',
            'guideline': 'Region: Each persona should have exactly ONE region from: Northeast, Southeast, Midwest, Southwest, West Coast, Pacific Northwest, etc. Vary regions across personas.'
        },
        'zip_codes': {
            'example': '["10001", "90210"]',
            'guideline': 'Zip codes: Realistic zip codes for their location'
        },
        'race': {
            'example': '["Asian"]',
            'guideline': 'Races: Include diverse ethnicities (Asian, Black/African American, Hispanic/Latino, White, Middle Eastern, Pacific Islander, Multiracial, etc.)'
        },
        'careers': {
            'example': '["Engineering"]',
            'guideline': 'Career: Each persona should have exactly ONE primary career from categories like: blue-collar, white-collar, creative, technical, service industry, healthcare, education, retail, etc. Vary across personas.'
        },
        'education': {
            'example': '["Bachelor\'s Degree"]',
            'guideline': 'Education: Each persona should have exactly ONE education level from: High School, Some College, College Degree, Graduate Degree. Vary across personas.'
        },
        'income_level': {
            'example': '["$75k-$100k"]',
            'guideline': 'Income: Each persona should have exactly ONE income range from: Under $25k, $25k-$50k, $50k-$75k, $75k-$100k, $100k-$150k, $150k-$200k, Over $200k. Distribute varied incomes across personas.'
        },
        'household_count': {
            'example': '["2"]',
            'guideline': 'Household size: Each persona should have exactly ONE household size from: 1, 2, 3, 4, 5, 6+. Vary across personas.'
        },
        'household_type': {
            'example': '["Home"]',
            'guideline': 'Household type: Each persona should have exactly ONE housing type from: Owner, Renter, Apartment, Home, Condo, Townhouse. Vary across personas.'
        }
    }

    # Use selected dimensions or default to all
    if not selected_dimensions:
        selected_dimensions = list(demographic_fields.keys())

    # Build demographics JSON structure
    demographics_json = "{\n"
    for dim in selected_dimensions:
        if dim in demographic_fields:
            demographics_json += f'        "{dim}": {demographic_fields[dim]["example"]},\n'
    demographics_json = demographics_json.rstrip(',\n') + "\n      }"

    # Build diversity guidelines
    diversity_guidelines = "\n".join([
        f"- {demographic_fields[dim]['guideline']}"
        for dim in selected_dimensions
        if dim in demographic_fields
    ])

    # Build diversity requirement text
    dimension_labels = ', '.join([dim.replace('_', ' ') for dim in selected_dimensions])

    prompt = f"""You are a marketing expert specializing in audience segmentation. Generate {num_personas} diverse, distinct audience segments for a marketing campaign.

{campaign_details}

REQUIREMENTS:
1. Generate {num_personas} UNIQUE and DIVERSE audience segments
2. Each segment should represent a different demographic group
3. Include realistic and varied combinations of demographics
4. Segments should be relevant to the product/campaign
5. Ensure diversity across {dimension_labels}

IMPORTANT:
- DO NOT create specific individual characters or stories
- DO NOT use names like "Sarah" or "John"
- DO create GENERIC DESCRIPTIONS of audience types/segments
- Describe "people like this" as a GROUP, not as an individual person
- Focus on characteristics, behaviors, and preferences of the segment

OUTPUT FORMAT (JSON):
Return a JSON array with {num_personas} audience segments. Each segment must have this exact structure:

{{
  "personas": [
    {{
      "description": "<Generic description of this audience segment - what characterizes people in this group, their typical behaviors, preferences, and why they would be interested in this product. Write about 'they' or 'this group' not 'he/she'. 2-3 sentences.>",
      "demographics": {demographics_json},
      "ai_generated": true,
      "model_provider": "<provider>",
      "model_name": "<model>"
    }}
  ]
}}

DIVERSITY GUIDELINES:
{diversity_guidelines}

DESCRIPTION EXAMPLES (DO):
✓ "This segment consists of tech-savvy professionals who prioritize efficiency and innovation in their daily workflows. They actively seek tools that integrate seamlessly with their existing systems and value data-driven solutions."
✓ "These are budget-conscious families in suburban areas who look for practical, long-lasting products. They tend to research extensively before purchasing and appreciate clear value propositions."

DESCRIPTION EXAMPLES (DON'T):
✗ "Sarah is a 32-year-old software engineer living in San Francisco who loves coffee and coding."
✗ "John works as a marketing manager and enjoys hiking on weekends with his dog."

Make each segment represent a meaningful audience group with distinct characteristics.
"""

    return prompt

def generate_with_openai(prompt: str, model_name: str = 'gpt-4o') -> Optional[List[Dict[str, Any]]]:
    """Generate personas using OpenAI"""
    try:
        models_without_temp = ['gpt-5', 'o1', 'o1-mini', 'o1-preview', 'o3-mini']

        params = {
            "model": MODELS['openai'][model_name],
            "messages": [
                {"role": "system", "content": "You are a marketing expert specializing in customer persona development. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        if model_name not in models_without_temp:
            params["temperature"] = 0.8

        response = openai.chat.completions.create(**params)
        result = json.loads(response.choices[0].message.content)

        # Handle both array and object responses
        if isinstance(result, list):
            personas = result
        elif isinstance(result, dict):
            personas = result.get('personas', [])
        else:
            print(f"Unexpected response type from OpenAI: {type(result)}")
            return None

        # Add model info to each persona
        for persona in personas:
            persona['model_provider'] = 'openai'
            persona['model_name'] = model_name

        return personas
    except Exception as e:
        print(f"Error with OpenAI {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_with_anthropic(prompt: str, model_name: str = 'claude-sonnet-4.5') -> Optional[List[Dict[str, Any]]]:
    """Generate personas using Anthropic Claude"""
    try:
        response = anthropic_client.messages.create(
            model=MODELS['anthropic'][model_name],
            max_tokens=8192,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )

        content = response.content[0].text
        # Extract JSON from response
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            result = json.loads(content[start:end])

            # Handle both array and object responses
            if isinstance(result, list):
                personas = result
            elif isinstance(result, dict):
                personas = result.get('personas', [])
            else:
                print(f"Unexpected response type from Anthropic: {type(result)}")
                return None

            # Add model info to each persona
            for persona in personas:
                persona['model_provider'] = 'anthropic'
                persona['model_name'] = model_name

            return personas
        return None
    except Exception as e:
        print(f"Error with Anthropic {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_with_google(prompt: str, model_name: str = 'gemini-2.5-pro') -> Optional[List[Dict[str, Any]]]:
    """Generate personas using Google Gemini"""
    try:
        model = genai.GenerativeModel(MODELS['google'][model_name])
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                response_mime_type="application/json"
            )
        )
        result = json.loads(response.text)

        # Handle both array and object responses
        if isinstance(result, list):
            personas = result
        elif isinstance(result, dict):
            personas = result.get('personas', [])
        else:
            print(f"Unexpected response type from Google: {type(result)}")
            return None

        # Add model info to each persona
        for persona in personas:
            persona['model_provider'] = 'google'
            persona['model_name'] = model_name

        return personas
    except Exception as e:
        print(f"Error with Google {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

async def generate_personas_for_campaign(
    campaign: Optional[Dict[str, Any]],
    num_personas: int = 20,
    model_provider: str = 'anthropic',
    model_name: Optional[str] = None,
    selected_dimensions: Optional[List[str]] = None,
    distribution_description: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate personas for a campaign using specified AI model

    Args:
        campaign: Campaign object with details (optional)
        num_personas: Number of personas to generate (default: 20)
        model_provider: AI provider ('openai', 'anthropic', 'google')
        model_name: Specific model name (optional)
        selected_dimensions: List of demographic dimensions to generate (optional)
        distribution_description: Optional guidance on how to distribute personas (optional)

    Returns:
        List of generated persona dictionaries
    """
    campaign_name = campaign.get('name', 'Unknown') if campaign else 'General'
    print(f"Generating {num_personas} personas for campaign: {campaign_name}")
    print(f"Using provider: {model_provider}, model: {model_name or 'default'}")
    print(f"Selected dimensions: {selected_dimensions or 'all'}")
    print(f"Distribution guidance: {distribution_description or 'none'}")

    prompt = create_persona_generation_prompt(campaign, num_personas, selected_dimensions, distribution_description)

    personas = None

    if model_provider == 'openai':
        model_name = model_name or 'gpt-4o'
        personas = generate_with_openai(prompt, model_name)
    elif model_provider == 'anthropic':
        model_name = model_name or 'claude-sonnet-4.5'
        personas = generate_with_anthropic(prompt, model_name)
    elif model_provider == 'google':
        model_name = model_name or 'gemini-2.5-pro'
        personas = generate_with_google(prompt, model_name)
    else:
        raise ValueError(f"Unknown provider: {model_provider}")

    if not personas:
        raise Exception(f"Failed to generate personas with {model_provider}")

    print(f"Successfully generated {len(personas)} personas")

    return personas

def main():
    """Test persona generation"""
    import argparse
    from pymongo import MongoClient

    parser = argparse.ArgumentParser(description='Generate personas for a campaign')
    parser.add_argument('--campaign-id', required=True, help='Campaign ID')
    parser.add_argument('--num-personas', type=int, default=20, help='Number of personas to generate')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'google'], default='anthropic',
                       help='AI provider to use')
    parser.add_argument('--model', help='Specific model name')

    args = parser.parse_args()

    # Connect to MongoDB
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]

    # Get campaign
    from bson import ObjectId
    campaign = db.campaigns.find_one({"_id": ObjectId(args.campaign_id)})

    if not campaign:
        print(f"Campaign {args.campaign_id} not found!")
        return

    # Generate personas
    import asyncio
    personas = asyncio.run(generate_personas_for_campaign(
        campaign=campaign,
        num_personas=args.num_personas,
        model_provider=args.provider,
        model_name=args.model
    ))

    # Save to database
    for persona in personas:
        persona['campaign_id'] = args.campaign_id
        from datetime import datetime
        persona['created_at'] = datetime.utcnow()
        persona['updated_at'] = datetime.utcnow()

    result = db.personas.insert_many(personas)
    print(f"\nSaved {len(result.inserted_ids)} personas to database")

if __name__ == '__main__':
    main()
