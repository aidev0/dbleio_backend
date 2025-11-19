# Database Setup

## Overview

The video marketing simulation uses MongoDB for data storage with the following architecture:

- **MongoDB**: Stores data in `snake_case` format
- **Backend (Python)**: Seeds and manages data in `snake_case`
- **Frontend (JavaScript/TypeScript)**: Uses `camelCase` with automatic conversion

## MongoDB Connection

Add your MongoDB connection string to `.env`:

```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/video_marketing_db?retryWrites=true&w=majority
```

## Collections

The database contains the following collections:

1. **campaigns** - Marketing campaigns
2. **personas** - Customer personas (20 per campaign)
3. **videos** - Test videos (4 per campaign)
4. **video_understandings** - AI analysis of videos
5. **llm_models** - AI model configurations (OpenAI, Anthropic, Google)
6. **marketing_simulation_results** - Simulation results for each persona/model
7. **marketing_simulation_summaries** - Aggregate simulation summaries
8. **feedbacks** - User feedback on videos
9. **synthesis_videos** - AI-generated combined videos

## Seeding the Database

### Prerequisites

Make sure pymongo is installed:

```bash
pip install pymongo
```

### Run the Seed Script

```bash
cd backend
./venv/bin/python3 seed_database.py
```

This will:
1. Clear all existing data
2. Create a demo campaign ("Half Price Drapes - Q4 2025")
3. Generate 20 customer personas
4. Create 4 test videos with analysis from `data/video*.json` files
5. Add 13 LLM models from `ai_agent.py`
6. Generate sample simulation results
7. Create sample feedbacks and synthesis video
8. Set up database indexes

### Output

```
================================================================================
SEEDING MONGODB DATABASE
================================================================================
Clearing all collections...
✓ All collections cleared

Seeding campaign...
✓ Campaign created: Half Price Drapes - Q4 2025 (ID: 6913d0e96e39d50fbc33abed)

Seeding personas...
✓ Created 20 personas

Seeding videos...
✓ Created 4 videos

...

✓ DATABASE SEEDING COMPLETE!
```

## Data Format

### MongoDB (snake_case)

```json
{
  "_id": ObjectId("..."),
  "campaign_id": "6913d0e96e39d50fbc33abed",
  "created_at": ISODate("2025-11-11T23:00:00.000Z"),
  "target_audience": "New Customers"
}
```

### Frontend JavaScript (camelCase)

```javascript
{
  _id: "...",
  campaignId: "6913d0e96e39d50fbc33abed",
  createdAt: "2025-11-11T23:00:00.000Z",
  targetAudience: "New Customers"
}
```

The API automatically converts between formats using `convertKeysToCamel()` utility.

## Video Analysis Data

Video understandings are populated from backend JSON files:

- `backend/data/video1.json` → Video 1 analysis
- `backend/data/video2.json` → Video 2 analysis
- `backend/data/video3.json` → Video 3 analysis
- `backend/data/video4.json` → Video 4 analysis

Each file contains:
- `summary`: Description of video content
- `objects`: List of objects shown
- `colors`: Color palette
- `texture`/`textures`: Material types
- `number_of_scene_cut`: Scene transitions
- `qualities_demonstrated`: Features highlighted
- `duration`: Video length
- `timestamp_most_important_info_shown`: Key moments

## LLM Models

Models are loaded from `ai_agent.py` MODELS dictionary:

**OpenAI**: gpt-5, gpt-4o, gpt-4o-mini, o1

**Anthropic**: claude-sonnet-4.5, claude-haiku-4.5, claude-opus-4.1, claude-3.7-sonnet

**Google**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash, gemini-2.0-flash-lite

## Indexes

The following indexes are created for performance:

- `campaigns.created_at` (descending)
- `personas.campaign_id`
- `videos.campaign_id`
- `video_understandings.campaign_id`
- `llm_models.provider + model_name`
- `marketing_simulation_results.persona_id`
- `feedbacks.campaign_id`
- `synthesis_videos.campaign_id`

## Troubleshooting

### Connection Issues

If you get connection errors:
1. Check your `MONGODB_URI` in `.env`
2. Verify network access in MongoDB Atlas
3. Ensure IP address is whitelisted

### Missing Data

Re-run the seed script to regenerate all data:

```bash
./venv/bin/python3 seed_database.py
```

This will clear and recreate all collections.
