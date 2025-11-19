#!/usr/bin/env python3
"""
Seed MongoDB database with demo data for the video marketing simulation
Uses actual JSON data from frontend/public/data directory
"""

import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in environment variables")

MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'video_marketing_db')

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Collection names
COLLECTIONS = {
    'campaigns': 'campaigns',
    'personas': 'personas',
    'videos': 'videos',
    'video_understandings': 'video_understandings',
    'llm_models': 'llm_models',
    'marketing_simulation_results': 'marketing_simulation_results',
    'marketing_simulation_summaries': 'marketing_simulation_summaries',
    'feedbacks': 'feedbacks',
    'synthesis_videos': 'synthesis_videos'
}

# Paths to frontend data
FRONTEND_DATA_DIR = Path(__file__).parent.parent / 'frontend' / 'public' / 'data'
BACKEND_DATA_DIR = Path(__file__).parent / 'data'

# LLM Models from ai_agent.py
LLM_MODELS = {
    'openai': {
        'gpt-5': 'gpt-5',
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini',
        'o1': 'o1'
    },
    'anthropic': {
        'claude-sonnet-4.5': 'claude-sonnet-4-5-20250929',
        'claude-haiku-4.5': 'claude-haiku-4-5-20251001',
        'claude-opus-4.1': 'claude-opus-4-1-20250805',
        'claude-3.7-sonnet': 'claude-3-7-sonnet-20250219'
    },
    'google': {
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
        'gemini-2.0-flash': 'gemini-2.0-flash',
        'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite'
    }
}

def clear_all_collections():
    """Delete all data from all collections"""
    print("Clearing all collections...")
    for collection_name in COLLECTIONS.values():
        db[collection_name].delete_many({})
    print("✓ All collections cleared")

def get_hardcoded_personas():
    """Return hardcoded persona data - all 20 personas with statistical fields"""
    return [
        {"id": 1, "name": "Persona 1", "demographics": {"gender": "Male", "age_mean": 63.0, "age_std": 6.982714934301164, "num_orders_mean": 3.5680870581262933, "num_orders_std": 2.449227500681923, "revenue_per_customer_mean": 756.5351737411357, "revenue_per_customer_std": 161.4581882133978, "region": "Northwest", "weight": 0.05}},
        {"id": 2, "name": "Persona 2", "demographics": {"gender": "Female", "age_mean": 35.0, "age_std": 7.330880728874676, "num_orders_mean": 9.415610164404923, "num_orders_std": 2.2701814444901136, "revenue_per_customer_mean": 122.6429437253827, "revenue_per_customer_std": 292.4774630404986, "region": "West", "weight": 0.05}},
        {"id": 3, "name": "Persona 3", "demographics": {"gender": "Male", "age_mean": 62.0, "age_std": 3.0038938292050714, "num_orders_mean": 14.890961830077046, "num_orders_std": 2.043703774069291, "revenue_per_customer_mean": 772.818476537109, "revenue_per_customer_std": 51.76657630492935, "region": "South", "weight": 0.049999999999999996}},
        {"id": 4, "name": "Persona 4", "demographics": {"gender": "Female", "age_mean": 51.0, "age_std": 6.059264473611897, "num_orders_mean": 2.9529140491285855, "num_orders_std": 1.2303616213380453, "revenue_per_customer_mean": 502.99802762306086, "revenue_per_customer_std": 164.01749605425897, "region": "Midwest", "weight": 0.049999999999999996}},
        {"id": 5, "name": "Persona 5", "demographics": {"gender": "Male", "age_mean": 68.0, "age_std": 5.571172192068058, "num_orders_mean": 9.293803964068594, "num_orders_std": 0.6161260317999944, "revenue_per_customer_mean": 768.2993370915822, "revenue_per_customer_std": 92.63103092182288, "region": "East", "weight": 0.05}},
        {"id": 6, "name": "Persona 6", "demographics": {"gender": "Female", "age_mean": 28.0, "age_std": 7.711008778424263, "num_orders_mean": 8.88603504983755, "num_orders_std": 1.4635412563497903, "revenue_per_customer_mean": 117.56287744223562, "revenue_per_customer_std": 107.72345640553725, "region": "Northeast", "weight": 0.05}},
        {"id": 7, "name": "Persona 7", "demographics": {"gender": "Male", "age_mean": 31.0, "age_std": 6.049983288913104, "num_orders_mean": 12.6647287643063, "num_orders_std": 0.9334116337694303, "revenue_per_customer_mean": 530.1666683305649, "revenue_per_customer_std": 95.55902194701558, "region": "Northwest", "weight": 0.05}},
        {"id": 8, "name": "Persona 8", "demographics": {"gender": "Female", "age_mean": 30.0, "age_std": 4.0397083143409445, "num_orders_mean": 8.94780458947988, "num_orders_std": 0.5782832311388965, "revenue_per_customer_mean": 1026.5132520544983, "revenue_per_customer_std": 162.43853334244142, "region": "West", "weight": 0.05}},
        {"id": 9, "name": "Persona 9", "demographics": {"gender": "Male", "age_mean": 60.0, "age_std": 5.989499894055426, "num_orders_mean": 13.906239290323636, "num_orders_std": 0.7212312551297988, "revenue_per_customer_mean": 315.58114866105973, "revenue_per_customer_std": 61.30682222763452, "region": "South", "weight": 0.05}},
        {"id": 10, "name": "Persona 10", "demographics": {"gender": "Female", "age_mean": 40.0, "age_std": 6.736600550686904, "num_orders_mean": 8.555689853447117, "num_orders_std": 1.9668779141596207, "revenue_per_customer_mean": 1161.7808379905518, "revenue_per_customer_std": 201.75856192167117, "region": "Midwest", "weight": 0.05}},
        {"id": 11, "name": "Persona 11", "demographics": {"gender": "Male", "age_mean": 53.0, "age_std": 7.010984903770199, "num_orders_mean": 2.0437090115167917, "num_orders_std": 2.9672173415012932, "revenue_per_customer_mean": 949.4692462263232, "revenue_per_customer_std": 99.6789203835431, "region": "East", "weight": 0.05}},
        {"id": 12, "name": "Persona 12", "demographics": {"gender": "Female", "age_mean": 35.0, "age_std": 3.9942120204440257, "num_orders_mean": 10.9587873384811, "num_orders_std": 2.475438851328014, "revenue_per_customer_mean": 766.5559722591125, "revenue_per_customer_std": 281.57521962833727, "region": "Northeast", "weight": 0.05}},
        {"id": 13, "name": "Persona 13", "demographics": {"gender": "Male", "age_mean": 52.0, "age_std": 7.315517129377968, "num_orders_mean": 9.72617377558581, "num_orders_std": 1.327245062131623, "revenue_per_customer_mean": 169.914185314626, "revenue_per_customer_std": 127.74558042891555, "region": "Northwest", "weight": 0.05}},
        {"id": 14, "name": "Persona 14", "demographics": {"gender": "Female", "age_mean": 59.0, "age_std": 5.9564889385386355, "num_orders_mean": 4.846105101860898, "num_orders_std": 1.903108564619253, "revenue_per_customer_mean": 521.2195622291688, "revenue_per_customer_std": 292.92802384727594, "region": "West", "weight": 0.05}},
        {"id": 15, "name": "Persona 15", "demographics": {"gender": "Male", "age_mean": 29.0, "age_std": 5.806385987847481, "num_orders_mean": 11.793540519363853, "num_orders_std": 1.7344889909109769, "revenue_per_customer_mean": 675.0061123201934, "revenue_per_customer_std": 156.8852545896374, "region": "South", "weight": 0.05}},
        {"id": 16, "name": "Persona 16", "demographics": {"gender": "Female", "age_mean": 37.0, "age_std": 4.008596011676981, "num_orders_mean": 13.540690339429272, "num_orders_std": 1.6884255579552794, "revenue_per_customer_mean": 719.603129174022, "revenue_per_customer_std": 223.87902160653186, "region": "Midwest", "weight": 0.05}},
        {"id": 17, "name": "Persona 17", "demographics": {"gender": "Male", "age_mean": 63.0, "age_std": 5.051914615178148, "num_orders_mean": 11.577715939602681, "num_orders_std": 1.0719954137290562, "revenue_per_customer_mean": 184.6779008116723, "revenue_per_customer_std": 122.43786322844201, "region": "East", "weight": 0.05}},
        {"id": 18, "name": "Persona 18", "demographics": {"gender": "Female", "age_mean": 26.0, "age_std": 6.121770240668966, "num_orders_mean": 5.138871601727995, "num_orders_std": 0.7637356495756765, "revenue_per_customer_mean": 602.1880275312012, "revenue_per_customer_std": 104.6101093042084, "region": "Northeast", "weight": 0.05}},
        {"id": 19, "name": "Persona 19", "demographics": {"gender": "Male", "age_mean": 35.0, "age_std": 4.6217251050263695, "num_orders_mean": 2.709231365809427, "num_orders_std": 1.3907445951924373, "revenue_per_customer_mean": 1097.5112857003294, "revenue_per_customer_std": 118.03306234615883, "region": "Northwest", "weight": 0.05}},
        {"id": 20, "name": "Persona 20", "demographics": {"gender": "Female", "age_mean": 37.0, "age_std": 7.303652916281717, "num_orders_mean": 1.0973298274366698, "num_orders_std": 1.7768682564439144, "revenue_per_customer_mean": 559.1521034636569, "revenue_per_customer_std": 105.52695261768257, "region": "West", "weight": 0.05}},
    ]

def load_summary():
    """Load summary from frontend JSON"""
    summary_file = FRONTEND_DATA_DIR / 'customers' / 'summary.json'
    with open(summary_file, 'r') as f:
        return json.load(f)

def load_video_analysis(video_num):
    """Load video analysis from backend JSON file"""
    video_file = BACKEND_DATA_DIR / f'video{video_num}.json'
    with open(video_file, 'r') as f:
        return json.load(f)

def load_evaluation(provider, model_name):
    """Load evaluation results from frontend JSON"""
    eval_file = FRONTEND_DATA_DIR / 'evaluations' / f'{provider}_{model_name}_evaluations.json'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            return json.load(f)
    return []

def seed_campaign():
    """Create demo campaign with flexible details structure"""
    print("\nSeeding campaign...")

    campaign = {
        'name': 'Half Price Drapes - Q4 2025',
        'platform': 'Facebook Ads',
        'target_audience': 'New Customers',
        'budget': 85000,
        'details': {
            'description': 'Facebook Ads • New Customer Acquisition • Window Treatments',
            'status': 'active',
            'products': 'Window Treatments',
            'goals': {
                'primary': 'Increase video ad conversions by 25% compared to current baseline',
                'secondary': 'Reduce cost per acquisition (CPA) by 15% through optimized video content',
                'ai_edited': 'Create an optimized video by combining the best-performing moments from all 4 test videos'
            }
        },
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }

    result = db[COLLECTIONS['campaigns']].insert_one(campaign)
    campaign_id = str(result.inserted_id)

    print(f"✓ Campaign created: {campaign['name']} (ID: {campaign_id})")
    return campaign_id

def seed_personas(campaign_id):
    """Create sample personas in NEW format with arrays"""
    print("\nSeeding personas...")

    # Get hardcoded persona data
    old_personas_data = get_hardcoded_personas()
    personas = []

    for old_persona in old_personas_data:  # Load all 20 personas
        # Convert old format to new format
        demographics = old_persona['demographics']

        # Determine age range from age_mean
        age_mean = demographics.get('age_mean', 30)
        if age_mean < 25:
            age_range = ['18-24']
        elif age_mean < 35:
            age_range = ['25-34']
        elif age_mean < 45:
            age_range = ['35-44']
        elif age_mean < 55:
            age_range = ['45-54']
        elif age_mean < 65:
            age_range = ['55-64']
        else:
            age_range = ['65+']

        persona = {
            'campaign_id': campaign_id,
            'name': old_persona['name'],
            'description': f"Customer persona based on historical data. Average age: {demographics.get('age_mean', 0):.0f}, Region: {demographics.get('region', 'Unknown')}",
            'demographics': {
                'age': age_range,
                'gender': [demographics.get('gender', 'Unknown')],
                'region': [demographics.get('region', 'Unknown')],
                'race': [],
                'careers': [],
                'education': [],
                'locations': [],
                'country': [],
                'zip_codes': [],
                'income_level': [],
                'household_count': [],
                'household_type': [],
                'custom_fields': {}
            },
            'ai_generated': False,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        personas.append(persona)

    if personas:
        try:
            result = db[COLLECTIONS['personas']].insert_many(personas)
            print(f"✓ Created {len(personas)} personas in new format")
            print(f"  Inserted IDs: {[str(id) for id in result.inserted_ids[:2]]}...")  # Show first 2
        except Exception as e:
            print(f"✗ Error inserting personas: {e}")
            raise
    else:
        print("⚠ No personas to insert")
    return personas

def seed_videos(campaign_id):
    """Create 4 test videos"""
    print("\nSeeding videos...")

    videos = [
        {
            'campaign_id': campaign_id,
            'title': 'Video 1',
            'url': '/data/videos/hpd_video_1.mov',
            'thumbnail_url': '/thumbnails/video1.jpg',
            'duration': 14,
            'created_at': datetime.utcnow()
        },
        {
            'campaign_id': campaign_id,
            'title': 'Video 2',
            'url': '/data/videos/hpd_video_2.mov',
            'thumbnail_url': '/thumbnails/video2.jpg',
            'duration': 30,
            'created_at': datetime.utcnow()
        },
        {
            'campaign_id': campaign_id,
            'title': 'Video 3',
            'url': '/data/videos/hpd_video_3.mov',
            'thumbnail_url': '/thumbnails/video3.jpg',
            'duration': 14,
            'created_at': datetime.utcnow()
        },
        {
            'campaign_id': campaign_id,
            'title': 'Video 4',
            'url': '/data/videos/hpd_video_4.mov',
            'thumbnail_url': '/thumbnails/video4.jpg',
            'duration': 18,
            'created_at': datetime.utcnow()
        }
    ]

    result = db[COLLECTIONS['videos']].insert_many(videos)
    video_ids = result.inserted_ids  # Keep as ObjectIds, don't convert to string

    print(f"✓ Created {len(videos)} videos")
    return video_ids

def seed_video_understandings(campaign_id, video_ids):
    """Create video understandings from backend JSON files"""
    print("\nSeeding video understandings...")

    video_understandings = []

    for i, video_id in enumerate(video_ids):
        analysis = load_video_analysis(i + 1)

        understanding = {
            'video_id': video_id,
            'campaign_id': campaign_id,
            'summary': analysis['summary'],
            'objects': analysis['objects'],
            'colors': analysis['colors'],
            'created_at': datetime.utcnow()
        }

        # Add optional fields if they exist
        if 'texture' in analysis:
            understanding['texture'] = analysis['texture']
        if 'textures' in analysis:
            understanding['textures'] = analysis['textures']
        if 'number_of_scene_cut' in analysis:
            understanding['number_of_scene_cut'] = analysis['number_of_scene_cut']
        if 'qualities_demonstrated' in analysis:
            understanding['qualities_demonstrated'] = analysis['qualities_demonstrated']
        if 'duration' in analysis:
            understanding['duration'] = analysis['duration']
        if 'timestamp_most_important_info_shown' in analysis:
            understanding['timestamp_most_important_info_shown'] = analysis['timestamp_most_important_info_shown']

        video_understandings.append(understanding)

    db[COLLECTIONS['video_understandings']].insert_many(video_understandings)
    print(f"✓ Created {len(video_understandings)} video understandings from backend data")

def seed_llm_models():
    """Create LLM model entries"""
    print("\nSeeding LLM models...")

    models = []
    for provider, provider_models in LLM_MODELS.items():
        for model_name, model_full_name in provider_models.items():
            model = {
                'provider': provider,
                'model_name': model_name,
                'model_full_name': model_full_name,
                'created_at': datetime.utcnow()
            }
            models.append(model)

    db[COLLECTIONS['llm_models']].insert_many(models)
    print(f"✓ Created {len(models)} LLM models")
    return models

def seed_simulation_results():
    """Create simulation results from frontend evaluation JSON files"""
    print("\nSeeding simulation results from frontend evaluation files...")

    all_results = []

    # Load evaluations for each model that has a file
    evaluation_files = {
        ('openai', 'gpt-5'): 'openai_gpt-5_evaluations.json',
        ('anthropic', 'claude-sonnet-4.5'): 'anthropic_claude-sonnet-4.5_evaluations.json',
        ('google', 'gemini-2.5-pro'): 'google_gemini-2.5-pro_evaluations.json',
    }

    for (provider, model_name), filename in evaluation_files.items():
        eval_file = FRONTEND_DATA_DIR / 'evaluations' / filename
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                evaluations = json.load(f)

            for eval_data in evaluations:
                result = {
                    'persona_id': eval_data['persona_id'],
                    'persona_name': eval_data['persona_name'],
                    'provider': eval_data['provider'],
                    'model': eval_data['model'],
                    'evaluation': eval_data['evaluation'],
                    'created_at': datetime.utcnow()
                }
                all_results.append(result)

            print(f"  ✓ Loaded {len(evaluations)} evaluations for {provider}/{model_name}")

    if all_results:
        db[COLLECTIONS['marketing_simulation_results']].insert_many(all_results)
        print(f"✓ Created {len(all_results)} simulation results from frontend data")

    return all_results

def seed_feedbacks(campaign_id, video_ids):
    """Create detailed video analysis feedbacks"""
    print("\nSeeding feedbacks...")

    feedbacks = [
        {
            'campaign_id': campaign_id,
            'video_id': video_ids[0],
            'video_number': 1,
            'title': 'Clear & Concise',
            'strengths': [
                'Clear, concise demonstration (14 seconds)',
                'Effective room darkening/light filtering demo',
                'Neutral oatmeal tone - universally appealing',
                'Simple, trustworthy presentation',
                'Fast-loading, mobile-friendly format'
            ],
            'weaknesses': [
                'Too short - lacks feature depth',
                'No installation/header type details',
                'Missing quality indicators (weighted hem, etc.)',
                'Single room/color limits versatility perception',
                'No clear call-to-action'
            ],
            'created_at': datetime.utcnow()
        },
        {
            'campaign_id': campaign_id,
            'video_id': video_ids[1],
            'video_number': 2,
            'title': 'Feature-Rich & Professional',
            'strengths': [
                'Comprehensive feature coverage (30 seconds)',
                'Multiple header types shown (hook belt, back tabs, pole pocket)',
                'Quality cues: weighted hem, room darkening tech',
                'Calm pacing (4 cuts) - easy to process',
                'Professional staging with neutral palette',
                'Appeals to high-value, detail-oriented buyers'
            ],
            'weaknesses': [
                'Leans contemporary vs. mid-century modern',
                'Too technical - over-indexes on specs vs. lifestyle',
                'May feel less accessible to budget-conscious buyers',
                'Lacks emotional warmth/connection',
                'No promotional hook or urgency',
                'Doesn\'t show light transformation dynamically'
            ],
            'created_at': datetime.utcnow()
        },
        {
            'campaign_id': campaign_id,
            'video_id': video_ids[2],
            'video_number': 3,
            'title': 'Variety Showcase',
            'strengths': [
                'Shows extensive color/texture variety',
                'Demonstrates product range breadth',
                'Multiple room settings shown',
                'High energy, attention-grabbing'
            ],
            'weaknesses': [
                'Way too fast (14 cuts in 14 seconds)',
                'Cognitive overload - can\'t process information',
                'Feels promotional vs. informative',
                'Color variety overwhelms instead of inspires',
                'No functional details or quality indicators',
                'Lacks cohesive aesthetic/brand identity',
                'Won\'t build purchase confidence'
            ],
            'created_at': datetime.utcnow()
        },
        {
            'campaign_id': campaign_id,
            'video_id': video_ids[3],
            'video_number': 4,
            'title': 'Lifestyle & Aspiration',
            'strengths': [
                'Beautiful emotional opening with natural light',
                'Lifestyle integration - shows lived-in spaces',
                'Good pacing (4.5s per cut)',
                'Light transformation is dramatic and clear',
                'Strong call-to-action at end',
                'Appeals to aspirational/younger buyers'
            ],
            'weaknesses': [
                'Too short (18 seconds) - incomplete story',
                'Light on technical specifications',
                'Missing installation/header details',
                'Lacks quality indicators for confident purchase',
                'More promotional than educational'
            ],
            'created_at': datetime.utcnow()
        }
    ]

    db[COLLECTIONS['feedbacks']].insert_many(feedbacks)
    print(f"✓ Created {len(feedbacks)} video analysis feedbacks")

def seed_synthesis_video(campaign_id, video_ids):
    """Create sample synthesis video with accurate timeline from VIDEO_EDITING_README.md"""
    print("\nSeeding synthesis video...")

    from datetime import timedelta
    now = datetime.utcnow()

    synthesis_video = {
        'campaign_id': campaign_id,
        'title': 'AI-Optimized Marketing Video',
        'description': 'Synthesized video combining best-performing segments from all 4 test videos based on persona preferences',
        'source_videos': video_ids,
        'timeline': [
            {
                'output_start': 0,
                'output_end': 5,
                'source_video_num': 4,
                'video_id': video_ids[3],
                'source_start': 0,
                'source_end': 5,
                'duration': 5,
                'description': 'Video 4 opening (emotional hook)'
            },
            {
                'output_start': 5,
                'output_end': 7,
                'source_video_num': 1,
                'video_id': video_ids[0],
                'source_start': 0,
                'source_end': 2,
                'duration': 2,
                'description': 'Video 1 (clear demonstration)'
            },
            {
                'output_start': 7,
                'output_end': 9,
                'source_video_num': 2,
                'video_id': video_ids[1],
                'source_start': 1,
                'source_end': 3,
                'duration': 2,
                'description': 'Video 2 (versatility)'
            },
            {
                'output_start': 9,
                'output_end': 11,
                'source_video_num': 3,
                'video_id': video_ids[2],
                'source_start': 1,
                'source_end': 3,
                'duration': 2,
                'description': 'Video 3 (multiple rooms)'
            },
            {
                'output_start': 11,
                'output_end': 13,
                'source_video_num': 2,
                'video_id': video_ids[1],
                'source_start': 10,
                'source_end': 12,
                'duration': 2,
                'description': 'Video 2 (feature showcase)'
            },
            {
                'output_start': 13,
                'output_end': 15,
                'source_video_num': 2,
                'video_id': video_ids[1],
                'source_start': 21,
                'source_end': 23,
                'duration': 2,
                'description': 'Video 2 (feature showcase)'
            },
            {
                'output_start': 15,
                'output_end': 17,
                'source_video_num': 2,
                'video_id': video_ids[1],
                'source_start': 19,
                'source_end': 21,
                'duration': 2,
                'description': 'Video 2 (feature showcase)'
            },
            {
                'output_start': 17,
                'output_end': 18,
                'source_video_num': 2,
                'video_id': video_ids[1],
                'source_start': 18,
                'source_end': 19,
                'duration': 1,
                'description': 'Video 2 (feature showcase)'
            },
            {
                'output_start': 18,
                'output_end': 21,
                'source_video_num': 4,
                'video_id': video_ids[3],
                'source_start': 13,
                'source_end': 16,
                'duration': 3,
                'description': 'Video 4 (transformation & CTA)'
            },
            {
                'output_start': 21,
                'output_end': 24,
                'source_video_num': 4,
                'video_id': video_ids[3],
                'source_start': 15,
                'source_end': 18,
                'duration': 3,
                'description': 'Video 4 (transformation & CTA)'
            }
        ],
        'total_duration': 24,
        'production_specifications': {
            'pacing_strategy': [
                '8-10 total cuts over 26-28 seconds',
                '3 seconds per scene (sweet spot)',
                'Faster than Video 2 (7.5s), slower than Video 3 (1s)',
                'Engaging but not frantic'
            ],
            'color_palette': [
                'Lead with neutrals: gray, oatmeal, white',
                'ONE accent (sage green or soft blue)',
                'Avoid Video 3\'s rainbow assault',
                'Universal appeal + versatility'
            ],
            'text_overlay_philosophy': [
                'Maximum 5 text moments total',
                '3-5 words per card',
                '70% visual demo, 30% text support',
                'Show, don\'t tell'
            ],
            'audio_strategy': [
                'Subtle instrumental - warm, modern',
                'Natural sound effects (fabric rustle)',
                'NO voiceover - universally accessible',
                'Music fades during feature section'
            ]
        },
        'synthesis_elements': [
            {
                'source': 'Video 1',
                'elements': 'Clear room darkening demo, neutral aesthetic'
            },
            {
                'source': 'Video 2',
                'elements': 'Feature depth, quality indicators, professional staging'
            },
            {
                'source': 'Video 3',
                'elements': 'Concept of variety (but executed properly with 3 colors, not 14)'
            },
            {
                'source': 'Video 4',
                'elements': 'Emotional opening, lifestyle integration, light transformation, strong CTA'
            }
        ],
        'edit_times': {
            'initial_generation': now - timedelta(minutes=30),
            'last_edit': now - timedelta(minutes=5),
            'version': 2
        },
        'status': 'draft',
        'output_url': '/data/videos/combined_marketing_video.mp4',
        'created_at': now,
        'updated_at': now
    }

    db[COLLECTIONS['synthesis_videos']].insert_one(synthesis_video)
    print("✓ Created synthesis video with accurate timeline (10 segments, 24 seconds)")

def create_indexes():
    """Create indexes for better performance"""
    print("\nCreating indexes...")

    db[COLLECTIONS['campaigns']].create_index([('created_at', -1)])
    db[COLLECTIONS['personas']].create_index([('campaign_id', 1)])
    db[COLLECTIONS['videos']].create_index([('campaign_id', 1)])
    db[COLLECTIONS['video_understandings']].create_index([('campaign_id', 1)])
    db[COLLECTIONS['llm_models']].create_index([('provider', 1), ('model_name', 1)])
    db[COLLECTIONS['marketing_simulation_results']].create_index([('persona_id', 1)])
    db[COLLECTIONS['feedbacks']].create_index([('campaign_id', 1)])
    db[COLLECTIONS['synthesis_videos']].create_index([('campaign_id', 1)])

    print("✓ Indexes created")

def main():
    """Main seeding function"""
    print("="*80)
    print("SEEDING MONGODB DATABASE WITH REAL MVP DATA")
    print("="*80)

    try:
        # Clear existing data
        clear_all_collections()

        # Seed data in order
        campaign_id = seed_campaign()
        personas = seed_personas(campaign_id)
        video_ids = seed_videos(campaign_id)
        seed_video_understandings(campaign_id, video_ids)
        models = seed_llm_models()
        results = seed_simulation_results()
        seed_feedbacks(campaign_id, video_ids)
        seed_synthesis_video(campaign_id, video_ids)

        # Create indexes
        create_indexes()

        print("\n" + "="*80)
        print("✓ DATABASE SEEDING COMPLETE!")
        print("="*80)
        print(f"\nSummary:")
        print(f"  Campaign ID: {campaign_id}")
        print(f"  Personas: {len(personas)} (hardcoded in seed_database.py)")
        print(f"  Videos: {len(video_ids)}")
        print(f"  LLM Models: {len(models)}")
        print(f"  Simulation Results: {len(results)} (from frontend/public/data/evaluations/*.json)")
        print(f"  Collections: {len(COLLECTIONS)}")

    except Exception as e:
        print(f"\n✗ Error seeding database: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        client.close()

if __name__ == '__main__':
    main()
