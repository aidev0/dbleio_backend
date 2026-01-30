#!/usr/bin/env python3
"""
Persona routes and models
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Create router
router = APIRouter(prefix="/api/personas", tags=["personas"])

# Pydantic Models
class Demographics(BaseModel):
    # New format - arrays for multi-select
    age: Optional[List[str]] = Field(default_factory=list)
    gender: Optional[List[str]] = Field(default_factory=list)
    locations: Optional[List[str]] = Field(default_factory=list)
    country: Optional[List[str]] = Field(default_factory=list)
    region: Optional[List[str]] = Field(default_factory=list)
    zip_codes: Optional[List[str]] = Field(default_factory=list)
    race: Optional[List[str]] = Field(default_factory=list)
    careers: Optional[List[str]] = Field(default_factory=list)
    education: Optional[List[str]] = Field(default_factory=list)
    income_level: Optional[List[str]] = Field(default_factory=list)
    household_count: Optional[List[str]] = Field(default_factory=list)
    household_type: Optional[List[str]] = Field(default_factory=list)
    custom_fields: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # Original demo format - statistical fields (optional)
    age_mean: Optional[float] = None
    age_std: Optional[float] = None
    num_orders_mean: Optional[float] = None
    num_orders_std: Optional[float] = None
    revenue_per_customer_mean: Optional[float] = None
    revenue_per_customer_std: Optional[float] = None
    weight: Optional[float] = None

class PersonaCreate(BaseModel):
    campaign_id: Optional[str] = None
    name: Optional[str] = None  # Made optional since we generate from demographics
    demographics: Demographics
    description: Optional[str] = None
    ai_generated: bool = False
    model_provider: Optional[str] = None
    model_name: Optional[str] = None

class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    demographics: Optional[Demographics] = None
    description: Optional[str] = None

class PersonaResponse(BaseModel):
    id: str
    campaign_id: Optional[str] = None
    name: str
    demographics: Demographics
    description: Optional[str] = None
    ai_generated: bool
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class PersonaGenerationRequest(BaseModel):
    campaign_id: Optional[str] = None
    num_personas: int = 20
    model_provider: str = "anthropic"
    model_name: Optional[str] = None
    selected_dimensions: Optional[List[str]] = None
    distribution_description: Optional[str] = None

# Helper functions
def persona_helper(persona) -> dict:
    """Convert MongoDB persona to dict"""
    # Generate display name from demographics if name doesn't exist
    name = persona.get("name")
    if not name:
        # Import the display name generator
        from ai_agent import generate_persona_display_name
        name = generate_persona_display_name(persona)

    return {
        "id": str(persona["_id"]),
        "campaign_id": persona.get("campaign_id"),
        "name": name,
        "demographics": persona.get("demographics", {}),
        "description": persona.get("description"),
        "ai_generated": persona.get("ai_generated", False),
        "model_provider": persona.get("model_provider"),
        "model_name": persona.get("model_name"),
        "created_at": persona.get("created_at", datetime.utcnow()),
        "updated_at": persona.get("updated_at", datetime.utcnow()),
    }

# Routes
@router.get("", response_model=List[PersonaResponse])
async def get_personas(campaign_id: Optional[str] = None):
    """Get all personas, optionally filtered by campaign_id"""
    try:
        query = {}
        if campaign_id:
            query["campaign_id"] = campaign_id

        personas = db.personas.find(query)
        return [persona_helper(persona) for persona in personas]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{persona_id}", response_model=PersonaResponse)
async def get_persona(persona_id: str):
    """Get a single persona by ID"""
    try:
        persona = db.personas.find_one({"_id": ObjectId(persona_id)})
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        return persona_helper(persona)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("", response_model=PersonaResponse, status_code=status.HTTP_201_CREATED)
async def create_persona(persona: PersonaCreate):
    """Create a new persona"""
    try:
        persona_dict = persona.model_dump()
        persona_dict["created_at"] = datetime.utcnow()
        persona_dict["updated_at"] = datetime.utcnow()

        result = db.personas.insert_one(persona_dict)
        new_persona = db.personas.find_one({"_id": result.inserted_id})

        return persona_helper(new_persona)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{persona_id}", response_model=PersonaResponse)
async def update_persona(persona_id: str, persona_update: PersonaUpdate):
    """Update an existing persona"""
    try:
        # Check if persona exists
        existing_persona = db.personas.find_one({"_id": ObjectId(persona_id)})
        if not existing_persona:
            raise HTTPException(status_code=404, detail="Persona not found")

        # Update fields
        update_dict = {k: v for k, v in persona_update.model_dump().items() if v is not None}
        update_dict["updated_at"] = datetime.utcnow()

        db.personas.update_one(
            {"_id": ObjectId(persona_id)},
            {"$set": update_dict}
        )

        updated_persona = db.personas.find_one({"_id": ObjectId(persona_id)})
        return persona_helper(updated_persona)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_persona(persona_id: str):
    """Delete a persona"""
    try:
        result = db.personas.delete_one({"_id": ObjectId(persona_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Persona not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_personas_background(
    task_id: str,
    campaign_id: Optional[str],
    num_personas: int,
    model_provider: str,
    model_name: Optional[str],
    selected_dimensions: Optional[List[str]],
    distribution_description: Optional[str]
):
    """Background task to generate personas with AI"""
    from task_manager import task_manager, TaskStatus
    from persona_generation import generate_personas_for_campaign

    try:
        print(f"Starting background persona generation for task {task_id}")
        task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=10, message="Preparing persona generation...")

        # Get campaign details if campaign_id is provided
        campaign = None
        if campaign_id:
            campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})

        task_manager.update_task(task_id, progress=20, message=f"Generating {num_personas} personas with AI...")

        # Generate personas
        personas = await generate_personas_for_campaign(
            campaign=campaign,
            num_personas=num_personas,
            model_provider=model_provider,
            model_name=model_name,
            selected_dimensions=selected_dimensions,
            distribution_description=distribution_description
        )

        task_manager.update_task(task_id, progress=80, message="Saving personas to database...")

        # Save to database
        saved_personas = []
        for i, persona_data in enumerate(personas):
            # Update progress as we save
            progress = 80 + int((i / len(personas)) * 15)
            task_manager.update_task(task_id, progress=progress, message=f"Saving persona {i+1}/{len(personas)}...")

            # Generate display name if not present
            if "name" not in persona_data or not persona_data["name"]:
                from ai_agent import generate_persona_display_name
                persona_data["name"] = generate_persona_display_name(persona_data)

            persona_data["campaign_id"] = campaign_id
            persona_data["created_at"] = datetime.utcnow()
            persona_data["updated_at"] = datetime.utcnow()

            result = db.personas.insert_one(persona_data)
            new_persona = db.personas.find_one({"_id": result.inserted_id})
            saved_personas.append(persona_helper(new_persona))

        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message=f"Successfully generated {len(saved_personas)} personas",
            result={"count": len(saved_personas), "persona_ids": [p["id"] for p in saved_personas]}
        )
        print(f"✓ Background persona generation completed for task {task_id}")
    except Exception as e:
        error_msg = f"Error generating personas: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        task_manager.update_task(task_id, status=TaskStatus.FAILED, message=error_msg, error=str(e))

@router.post("/generate")
async def generate_personas(request: PersonaGenerationRequest, background_tasks: BackgroundTasks):
    """Generate personas using AI in the background"""
    import uuid
    from task_manager import task_manager

    try:
        # Validate campaign exists if provided
        if request.campaign_id:
            campaign = db.campaigns.find_one({"_id": ObjectId(request.campaign_id)})
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")

        # Create background task
        task_id = f"persona_generation_{uuid.uuid4().hex[:8]}"
        task_manager.create_task(
            task_id,
            "persona_generation",
            metadata={
                "campaign_id": request.campaign_id,
                "num_personas": request.num_personas,
                "model_provider": request.model_provider
            }
        )

        # Schedule generation in background
        background_tasks.add_task(
            generate_personas_background,
            task_id,
            request.campaign_id,
            request.num_personas,
            request.model_provider,
            request.model_name,
            request.selected_dimensions,
            request.distribution_description
        )

        print(f"Scheduled background persona generation with task {task_id}")

        return {
            "success": True,
            "message": "Persona generation started",
            "task_id": task_id
        }
    except Exception as e:
        print(f"Error in generate_personas: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
