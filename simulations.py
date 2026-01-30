"""
Simulations API - Run video marketing simulations with different LLM models
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from bson import ObjectId
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import AI evaluation functions
from ai_agent import evaluate_persona, MODELS, generate_persona_display_name

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGODB_URI)
db = client['dble_db']
simulations_collection = db['simulations']

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=3)

# Router
router = APIRouter(prefix="/api/simulations", tags=["simulations"])

# Also create an evaluations router for getting evaluations
evaluations_router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


# Pydantic Models
class SimulationConfig(BaseModel):
    """Configuration for a simulation run"""
    name: str = Field(..., description="Name of the simulation")
    description: Optional[str] = Field(None, description="Description of the simulation")
    campaign_id: str = Field(..., description="Campaign ID to run simulation for")
    model_provider: Literal['openai', 'anthropic', 'google'] = Field(..., description="LLM provider to use")
    model_name: str = Field(..., description="Specific model name (e.g., 'gpt-5', 'claude-sonnet-4.5', 'gemini-2.5-pro')")
    persona_ids: Optional[List[str]] = Field(None, description="Specific persona IDs to evaluate (None for all)")
    video_ids: Optional[List[str]] = Field(None, description="Specific video IDs to evaluate (None for all)")


class SimulationCreate(BaseModel):
    """Request body for creating a simulation"""
    name: str
    description: Optional[str] = None
    campaign_id: str
    model_provider: Literal['openai', 'anthropic', 'google']
    model_name: str
    persona_ids: Optional[List[str]] = None
    video_ids: Optional[List[str]] = None
    auto_run: bool = Field(True, description="Automatically run simulation after creation")


class SimulationUpdate(BaseModel):
    """Request body for updating a simulation"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[Literal['pending', 'running', 'completed', 'failed', 'cancelled']] = None


class PersonaEvaluation(BaseModel):
    """Result of evaluating a single persona"""
    persona_id: str
    persona_name: str
    most_preferred_video: str
    preference_ranking: List[str]
    confidence_score: int
    video_opinions: Dict[str, str]
    reasoning: str
    semantic_analysis: Optional[str] = None


class SimulationResult(BaseModel):
    """Complete simulation results"""
    total_personas_evaluated: int
    total_videos: int
    evaluations: List[PersonaEvaluation]
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None


class Simulation(BaseModel):
    """Complete simulation object"""
    id: str = Field(alias="_id")
    name: str
    description: Optional[str] = None
    campaign_id: str
    model_provider: Literal['openai', 'anthropic', 'google']
    model_name: str
    persona_ids: Optional[List[str]] = None
    video_ids: Optional[List[str]] = None
    status: Literal['pending', 'running', 'completed', 'failed', 'cancelled'] = 'pending'
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[SimulationResult] = None

    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


# Helper Functions
def serialize_simulation(simulation_doc: Dict) -> Dict:
    """Convert MongoDB document to serializable dict"""
    if simulation_doc:
        simulation_doc['_id'] = str(simulation_doc['_id'])
        # Convert datetime objects
        for field in ['created_at', 'updated_at', 'started_at', 'completed_at']:
            if field in simulation_doc and simulation_doc[field]:
                if isinstance(simulation_doc[field], datetime):
                    simulation_doc[field] = simulation_doc[field].isoformat()
        # Convert results completion_time
        if simulation_doc.get('results') and simulation_doc['results'].get('completion_time'):
            if isinstance(simulation_doc['results']['completion_time'], datetime):
                simulation_doc['results']['completion_time'] = simulation_doc['results']['completion_time'].isoformat()
    return simulation_doc


async def run_simulation_task(simulation_id: str):
    """Background task to run the simulation"""
    try:
        # Update status to running
        simulations_collection.update_one(
            {'_id': ObjectId(simulation_id)},
            {
                '$set': {
                    'status': 'running',
                    'started_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
            }
        )

        # Get simulation config
        simulation = simulations_collection.find_one({'_id': ObjectId(simulation_id)})
        if not simulation:
            raise Exception("Simulation not found")

        # Load personas
        persona_ids = simulation.get('persona_ids')
        if persona_ids:
            # Convert string IDs to ObjectIds for MongoDB query
            persona_object_ids = [ObjectId(pid) for pid in persona_ids]
            personas = list(db['personas'].find({'_id': {'$in': persona_object_ids}}))
        else:
            campaign_id = simulation.get('campaign_id')
            personas = list(db['personas'].find({'campaign_id': campaign_id}))

        if not personas:
            raise Exception("No personas found for simulation")

        # Load videos
        video_ids = simulation.get('video_ids')
        if video_ids:
            videos = list(db['videos'].find({'_id': {'$in': [ObjectId(vid) for vid in video_ids]}}))
        else:
            campaign_id = simulation.get('campaign_id')
            videos = list(db['videos'].find({'campaign_id': campaign_id}))

        if not videos:
            raise Exception("No videos found for simulation")

        # Format videos and include video understanding/analysis data
        formatted_videos = []
        for video in videos:
            video_copy = dict(video)
            video_copy['_id'] = str(video['_id'])
            video_copy['video_id'] = str(video['_id'])  # Add video_id for backwards compatibility

            # Fetch ALL video understanding/analysis data for this video
            video_understandings = list(db['video_understandings'].find(
                {'video_id': video['_id']}
            ).sort('created_at', -1))  # Sort by newest first

            if video_understandings:
                # Include all video understanding records
                video_copy['analyses'] = []
                for understanding in video_understandings:
                    video_copy['analyses'].append({
                        'summary': understanding.get('summary'),
                        'timeline': understanding.get('timeline', []),
                        'qualities_demonstrated': understanding.get('qualities_demonstrated', []),
                        'objects': understanding.get('objects', []),
                        'transcript': understanding.get('transcript'),
                        'created_at': understanding.get('created_at').isoformat() if understanding.get('created_at') else None,
                    })
            else:
                video_copy['analyses'] = []

            formatted_videos.append(video_copy)

        # Run evaluations
        evaluations = []
        provider = simulation['model_provider']
        model_name = simulation['model_name']

        for persona in personas:
            try:
                # Format persona - just convert ObjectId to string and pass raw data
                formatted_persona = dict(persona)
                if '_id' in formatted_persona:
                    formatted_persona['_id'] = str(formatted_persona['_id'])
                # Ensure we have an 'id' field for backwards compatibility
                if 'id' not in formatted_persona and '_id' in persona:
                    formatted_persona['id'] = str(persona['_id'])

                # Run evaluation
                evaluation_result = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    evaluate_persona,
                    formatted_persona,
                    formatted_videos,
                    provider,
                    model_name
                )

                if evaluation_result:
                    evaluations.append({
                        'persona_id': formatted_persona['id'],
                        'persona_name': generate_persona_display_name(formatted_persona),
                        'most_preferred_video': str(evaluation_result.get('most_preferred_video', '')),
                        'preference_ranking': [str(v) for v in evaluation_result.get('preference_ranking', [])],
                        'confidence_score': evaluation_result.get('confidence_score', 0),
                        'video_opinions': evaluation_result.get('video_opinions', {}),
                        'reasoning': evaluation_result.get('reasoning', ''),
                        'semantic_analysis': evaluation_result.get('semantic_analysis')
                    })
            except Exception as e:
                persona_id = formatted_persona.get('id', persona.get('_id', 'unknown'))
                print(f"Error evaluating persona {persona_id}: {e}")
                continue

        # Create video mapping (number -> video details) for frontend display
        video_mapping = {}
        for idx, video in enumerate(formatted_videos, 1):
            video_mapping[str(idx)] = {
                'id': video.get('_id'),
                'title': video.get('title', f'Video {idx}'),
                'url': video.get('url', ''),
                'thumbnail_url': video.get('thumbnail_url', '')
            }

        # Update simulation with results
        results = {
            'total_personas_evaluated': len(evaluations),
            'total_videos': len(formatted_videos),
            'evaluations': evaluations,
            'video_mapping': video_mapping,
            'completion_time': datetime.utcnow()
        }

        simulations_collection.update_one(
            {'_id': ObjectId(simulation_id)},
            {
                '$set': {
                    'status': 'completed',
                    'completed_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow(),
                    'results': results
                }
            }
        )

        print(f"Simulation {simulation_id} completed successfully")

    except Exception as e:
        print(f"Error running simulation {simulation_id}: {e}")
        simulations_collection.update_one(
            {'_id': ObjectId(simulation_id)},
            {
                '$set': {
                    'status': 'failed',
                    'updated_at': datetime.utcnow(),
                    'results': {
                        'error_message': str(e),
                        'total_personas_evaluated': 0,
                        'total_videos': 0,
                        'evaluations': []
                    }
                }
            }
        )


# API Endpoints

@router.get("/", response_model=List[Simulation])
async def list_simulations(
    campaign_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """List all simulations with optional filters"""
    try:
        query = {}
        if campaign_id:
            query['campaign_id'] = campaign_id
        if status:
            query['status'] = status

        simulations = list(simulations_collection.find(query).sort('created_at', -1).limit(limit))
        return [serialize_simulation(sim) for sim in simulations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching simulations: {str(e)}")


@router.get("/{simulation_id}", response_model=Simulation)
async def get_simulation(simulation_id: str):
    """Get a specific simulation by ID"""
    try:
        simulation = simulations_collection.find_one({'_id': ObjectId(simulation_id)})
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")
        return serialize_simulation(simulation)
    except Exception as e:
        if "invalid ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid simulation ID format")
        raise HTTPException(status_code=500, detail=f"Error fetching simulation: {str(e)}")


@router.post("/", response_model=Simulation)
async def create_simulation(simulation: SimulationCreate, background_tasks: BackgroundTasks):
    """Create a new simulation"""
    try:
        # Validate model exists
        if simulation.model_provider not in MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model provider: {simulation.model_provider}")

        if simulation.model_name not in MODELS[simulation.model_provider]:
            available_models = list(MODELS[simulation.model_provider].keys())
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name '{simulation.model_name}' for provider '{simulation.model_provider}'. Available models: {available_models}"
            )

        # Validate campaign exists
        campaign = db['campaigns'].find_one({'_id': ObjectId(simulation.campaign_id)})
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Create simulation document
        simulation_doc = {
            'name': simulation.name,
            'description': simulation.description,
            'campaign_id': simulation.campaign_id,
            'model_provider': simulation.model_provider,
            'model_name': simulation.model_name,
            'persona_ids': simulation.persona_ids,
            'video_ids': simulation.video_ids,
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'started_at': None,
            'completed_at': None,
            'results': None
        }

        result = simulations_collection.insert_one(simulation_doc)
        simulation_id = str(result.inserted_id)

        # Run simulation in background if auto_run is True
        if simulation.auto_run:
            background_tasks.add_task(run_simulation_task, simulation_id)

        # Return created simulation
        created_simulation = simulations_collection.find_one({'_id': result.inserted_id})
        return serialize_simulation(created_simulation)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating simulation: {str(e)}")


@router.post("/{simulation_id}/run")
async def run_simulation(simulation_id: str, background_tasks: BackgroundTasks):
    """Run or re-run a simulation"""
    try:
        simulation = simulations_collection.find_one({'_id': ObjectId(simulation_id)})
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        # Check if already running
        if simulation['status'] == 'running':
            raise HTTPException(status_code=400, detail="Simulation is already running")

        # Reset status to pending
        simulations_collection.update_one(
            {'_id': ObjectId(simulation_id)},
            {
                '$set': {
                    'status': 'pending',
                    'updated_at': datetime.utcnow(),
                    'started_at': None,
                    'completed_at': None
                }
            }
        )

        # Run simulation in background
        background_tasks.add_task(run_simulation_task, simulation_id)

        return {"message": "Simulation started", "simulation_id": simulation_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running simulation: {str(e)}")


@router.patch("/{simulation_id}", response_model=Simulation)
async def update_simulation(simulation_id: str, update: SimulationUpdate):
    """Update a simulation (name, description, or status)"""
    try:
        simulation = simulations_collection.find_one({'_id': ObjectId(simulation_id)})
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        # Build update dict
        update_dict = {'updated_at': datetime.utcnow()}
        if update.name is not None:
            update_dict['name'] = update.name
        if update.description is not None:
            update_dict['description'] = update.description
        if update.status is not None:
            # Validate status transition
            current_status = simulation['status']
            if current_status == 'running' and update.status != 'cancelled':
                raise HTTPException(status_code=400, detail="Can only cancel a running simulation")
            update_dict['status'] = update.status

        # Update simulation
        simulations_collection.update_one(
            {'_id': ObjectId(simulation_id)},
            {'$set': update_dict}
        )

        # Return updated simulation
        updated_simulation = simulations_collection.find_one({'_id': ObjectId(simulation_id)})
        return serialize_simulation(updated_simulation)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating simulation: {str(e)}")


@router.delete("/{simulation_id}")
async def delete_simulation(simulation_id: str):
    """Delete a simulation"""
    try:
        simulation = simulations_collection.find_one({'_id': ObjectId(simulation_id)})
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        # Check if running
        if simulation['status'] == 'running':
            raise HTTPException(status_code=400, detail="Cannot delete a running simulation. Cancel it first.")

        # Delete simulation
        simulations_collection.delete_one({'_id': ObjectId(simulation_id)})

        return {"message": "Simulation deleted successfully", "simulation_id": simulation_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting simulation: {str(e)}")


@router.get("/models/available")
async def get_available_models():
    """Get list of available LLM models"""
    return {
        "models": MODELS,
        "providers": list(MODELS.keys())
    }


# Evaluations endpoints
@evaluations_router.get("/")
async def get_evaluations(campaign_id: Optional[str] = None):
    """Get all evaluations, optionally filtered by campaign"""
    try:
        # Check if evaluations collection exists
        evaluations_collection = db['evaluations']

        query = {}
        if campaign_id:
            query['campaign_id'] = campaign_id

        evaluations = list(evaluations_collection.find(query))

        # If no evaluations in collection, try getting from completed simulations
        if not evaluations and campaign_id:
            simulations = list(simulations_collection.find({
                'campaign_id': campaign_id,
                'status': 'completed'
            }))

            # Combine all evaluations from simulations
            all_evals = []
            for sim in simulations:
                if sim.get('results') and sim['results'].get('evaluations'):
                    for eval_data in sim['results']['evaluations']:
                        # Check if the data is already wrapped or flat
                        if 'evaluation' in eval_data:
                            # Already wrapped
                            all_evals.append({
                                'campaign_id': campaign_id,
                                'simulation_id': str(sim['_id']),
                                'provider': sim['model_provider'],
                                'model': sim['model_name'],
                                'persona_id': eval_data.get('persona_id'),
                                'persona_name': eval_data.get('persona_name'),
                                'evaluation': eval_data.get('evaluation')
                            })
                        else:
                            # Flat structure - wrap it
                            all_evals.append({
                                'campaign_id': campaign_id,
                                'simulation_id': str(sim['_id']),
                                'provider': sim['model_provider'],
                                'model': sim['model_name'],
                                'persona_id': eval_data.get('persona_id'),
                                'persona_name': eval_data.get('persona_name'),
                                'evaluation': {
                                    'most_preferred_video': eval_data.get('most_preferred_video'),
                                    'preference_ranking': eval_data.get('preference_ranking', []),
                                    'confidence_score': eval_data.get('confidence_score', 0),
                                    'video_opinions': eval_data.get('video_opinions', {}),
                                    'reasoning': eval_data.get('reasoning', ''),
                                    'semantic_analysis': eval_data.get('semantic_analysis', '')
                                }
                            })
            print(f"Loaded {len(all_evals)} evaluations from {len(simulations)} simulations")
            return all_evals

        # Serialize MongoDB documents
        for eval in evaluations:
            if '_id' in eval:
                eval['_id'] = str(eval['_id'])

        return evaluations

    except Exception as e:
        print(f"Error getting evaluations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting evaluations: {str(e)}")
