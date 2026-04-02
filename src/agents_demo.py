"""
Agents Demo API
"""

import os
import json
from datetime import timedelta
from fastapi import APIRouter, HTTPException
from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
GCS_BUCKET = os.getenv("GCS_BUCKET", "dble-input-videos")
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

router = APIRouter(prefix="/api/agents", tags=["agents-demo"])

_gcs = None
def _get_gcs():
    global _gcs
    if _gcs: return _gcs
    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not creds_json: return None
    from google.cloud import storage
    from google.oauth2 import service_account
    c = json.loads(creds_json)
    _gcs = storage.Client(credentials=service_account.Credentials.from_service_account_info(c), project=c.get("project_id"))
    return _gcs

def _sign(gs_uri):
    if not gs_uri: return ""
    try:
        g = _get_gcs()
        if not g: return gs_uri
        return g.bucket(GCS_BUCKET).blob(gs_uri.replace(f"gs://{GCS_BUCKET}/", "")).generate_signed_url(version="v4", expiration=timedelta(hours=2), method="GET")
    except: return gs_uri


# ── BRANDS ──

@router.get("/brands")
async def list_brands():
    brands = []
    for b in db.brands.find():
        bid = str(b["_id"])
        wf = db.content_workflows.find_one({"brand_id": bid})
        has_r = has_sb = has_sim = False
        if wf:
            wid = str(wf["_id"])
            ss = wf.get("config", {}).get("stage_settings", {})
            has_r = bool(ss.get("research", {}).get("brand_instagram"))
            has_sb = bool(db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "storyboard", "output_data.storyboards": {"$exists": True, "$ne": []}}))
            has_sim = bool(db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "simulation_testing", "output_data": {"$exists": True, "$ne": {}}}))
        brands.append({"id": bid, "name": b.get("name", "Unknown"), "has_research": has_r, "has_storyboard": has_sb, "has_simulation": has_sim})
    return brands


# ── RESEARCH ──

@router.get("/research/{brand_id}")
async def get_research(brand_id: str):
    wf = db.content_workflows.find_one({"brand_id": brand_id})
    if not wf: raise HTTPException(404, "No workflow")
    ss = wf.get("config", {}).get("stage_settings", {})
    research = ss.get("research", {})
    if not research: raise HTTPException(404, "No research data")

    brand_ig = research.get("brand_instagram", {})
    competitor_ig = research.get("competitor_instagram", {})

    for tp in brand_ig.get("top_performers", []):
        for f in tp.get("extracted_frames", []):
            if f.get("gs_uri"): f["signed_url"] = _sign(f["gs_uri"])

    competitors_out = {}
    for uname, d in competitor_ig.items():
        for tp in d.get("top_performers", []):
            for f in tp.get("extracted_frames", []):
                if f.get("gs_uri"): f["signed_url"] = _sign(f["gs_uri"])
        competitors_out[uname] = {**d, "top_performers": d.get("top_performers", [])[:8]}

    brand = db.brands.find_one({"_id": ObjectId(brand_id)})
    return {
        "brand_name": brand.get("name") if brand else "Unknown",
        "workflow_title": wf.get("title"),
        "brand_url_analysis": research.get("brand_url_analysis", {}),
        "brand_instagram": {**brand_ig, "top_performers": brand_ig.get("top_performers", [])[:12]},
        "competitor_instagram": competitors_out,
        "trends": research.get("trends", {}),
        "financial": research.get("financial", {}),
    }


# ── VIDEO GENERATION (grouped by content_id) ──

@router.get("/video/{brand_id}")
async def get_video_generation(brand_id: str):
    wf = db.content_workflows.find_one({"brand_id": brand_id})
    if not wf: raise HTTPException(404, "No workflow")
    wid = str(wf["_id"])
    ss = wf.get("config", {}).get("stage_settings", {})
    brand = db.brands.find_one({"_id": ObjectId(brand_id)})

    # Gather concepts per content_id
    concepts_by_cid = {}
    for pid, pd in ss.get("concepts", {}).get("pieces", {}).items():
        for c in pd.get("generated_concepts", []):
            concepts_by_cid[pid] = c

    # ── Hardcoded content pieces for the Stitch Fix workflow demo ──
    # Assets were overwritten across dates, so we manually define what belongs together.
    STITCHFIX_WID = "699e7e303eda3c34061943e8"
    if wid == STITCHFIX_WID:
        sb_node = db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "storyboard", "output_data.storyboards": {"$exists": True, "$ne": []}})
        vid_node = db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "video_generation", "output_data": {"$exists": True, "$ne": {}}})
        all_sbs = sb_node["output_data"]["storyboards"] if sb_node else []
        all_vars = vid_node["output_data"].get("variations", []) if vid_node else []

        # Office Interview (Mar 10) — storyboard[1], characters, scenes, veo-3.1 videos
        office_sb = all_sbs[1] if len(all_sbs) > 1 else None
        office_cid = "69aa167ab145749e6e42176f"
        if office_sb:
            for ch in office_sb.get("characters", []):
                if ch.get("gs_uri"): ch["image_url"] = _sign(ch["gs_uri"])
            # Only keep scenes that have matching videos
            office_vid_scenes = {v["scene_number"] for v in all_vars if v.get("content_id") == office_cid and v.get("type") == "scene" and v.get("scene_number")}
            office_sb["scenes"] = [s for s in office_sb.get("scenes", []) if s["scene_number"] in office_vid_scenes]
            for i, s in enumerate(office_sb["scenes"], 1):
                if s.get("gs_uri"): s["image_url"] = _sign(s["gs_uri"])
                s["scene_number"] = i
            # Remap video scene numbers
            sorted_orig = sorted(office_vid_scenes)
            remap = {old: new for new, old in enumerate(sorted_orig, 1)}

        office_vids = []
        for v in all_vars:
            if v.get("content_id") != office_cid: continue
            vurl = v.get("video_url", "")
            if v.get("gs_uri"): vurl = _sign(v["gs_uri"])
            sn = v.get("scene_number")
            if sn in remap: sn = remap[sn]
            office_vids.append({**v, "video_url": vurl, "scene_number": sn})

        # Style Quiz (Feb 25) — use Feb 25 snapshot from content_workflow_states for scenes
        quiz_cid = "69aa1678b145749e6e421763"
        quiz_concept = concepts_by_cid.get("d27f9dec-aa05-4b4d-a365-14db25d53a54") or concepts_by_cid.get(quiz_cid)

        # Load the Feb 25 snapshot storyboard for Style Quiz
        quiz_sb = None
        state_snap = db.content_workflow_states.find_one({"workflow_id": wid})
        if state_snap:
            snap_sbs = state_snap.get("state", {}).get("stage_outputs", {}).get("storyboard", {}).get("storyboards", [])
            for ssb in snap_sbs:
                if ssb.get("content_id") == quiz_cid:
                    quiz_sb = ssb
                    # Sign character images (these may be overwritten but try)
                    for ch in quiz_sb.get("characters", []):
                        if ch.get("gs_uri"): ch["image_url"] = _sign(ch["gs_uri"])
                        elif ch.get("image_url", "").startswith("gs://"): ch["image_url"] = _sign(ch["image_url"])
                    # Scene images were overwritten — clear them so UI doesn't show wrong images
                    for sc in quiz_sb.get("scenes", []):
                        sc["image_url"] = None
                        sc["gs_uri"] = None
                    break

        quiz_vids = []
        for v in all_vars:
            if v.get("content_id") != quiz_cid: continue
            vurl = v.get("video_url", "")
            if v.get("gs_uri"): vurl = _sign(v["gs_uri"])
            quiz_vids.append({**v, "video_url": vurl})

        content_pieces = [
            {
                "content_id": office_cid,
                "concept": concepts_by_cid.get(office_cid),
                "storyboard": office_sb,
                "videos": office_vids,
            },
            {
                "content_id": quiz_cid,
                "concept": quiz_concept,
                "storyboard": quiz_sb,  # From Feb 25 snapshot, scene images cleared (overwritten)
                "videos": quiz_vids,
            },
        ]
    else:
        # Generic path for other brands
        sb_node = db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "storyboard", "output_data.storyboards": {"$exists": True, "$ne": []}})
        vid_node = db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "video_generation", "output_data": {"$exists": True, "$ne": {}}})
        storyboards_by_cid = {}
        if sb_node:
            for sb in sb_node["output_data"]["storyboards"]:
                cid = sb.get("content_id")
                if not cid: continue
                for ch in sb.get("characters", []):
                    if ch.get("gs_uri"): ch["image_url"] = _sign(ch["gs_uri"])
                for sc in sb.get("scenes", []):
                    if sc.get("gs_uri"): sc["image_url"] = _sign(sc["gs_uri"])
                storyboards_by_cid[cid] = sb
        videos_by_cid = {}
        if vid_node:
            for v in vid_node["output_data"].get("variations", []):
                cid = v.get("content_id")
                if not cid: continue
                vurl = v.get("video_url", "")
                if v.get("gs_uri"): vurl = _sign(v["gs_uri"])
                videos_by_cid.setdefault(cid, []).append({**v, "video_url": vurl})
        all_cids = set(storyboards_by_cid.keys()) | set(videos_by_cid.keys())
        content_pieces = []
        for cid in all_cids:
            content_pieces.append({
                "content_id": cid,
                "concept": concepts_by_cid.get(cid),
                "storyboard": storyboards_by_cid.get(cid),
                "videos": videos_by_cid.get(cid, []),
            })

    return {
        "brand_name": brand.get("name") if brand else "Unknown",
        "workflow_title": wf.get("title"),
        "content_pieces": content_pieces,
    }


# ── SIMULATIONS ──

DEMO_SIM_BRANDS = {
    "half-price-drapes": {"name": "Half Price Drapes", "type": "campaign", "campaign_id": "691d2384aafb4c51925757f7"},
    "stitch-fix": {"name": "Stitch Fix", "type": "workflow", "brand_id": "699e7d000b1340eb9b44930f"},
}

@router.get("/simulations/brands")
async def list_simulation_brands():
    out = []
    for slug, info in DEMO_SIM_BRANDS.items():
        if info["type"] == "campaign":
            c = db.campaigns.find_one({"_id": ObjectId(info["campaign_id"])})
            cnt = db.simulations.count_documents({"campaign_id": info["campaign_id"], "status": "completed"})
            out.append({"slug": slug, "name": info["name"], "type": "campaign", "campaign_name": c.get("name") if c else None, "simulation_count": cnt})
        else:
            wf = db.content_workflows.find_one({"brand_id": info["brand_id"]})
            has = bool(wf and db.content_workflow_nodes.find_one({"workflow_id": str(wf["_id"]), "stage_key": "simulation_testing", "output_data": {"$exists": True, "$ne": {}}}))
            out.append({"slug": slug, "name": info["name"], "type": "workflow", "campaign_name": wf.get("title") if wf else None, "simulation_count": 1 if has else 0})
    return out

@router.get("/simulations/{slug}")
async def get_simulation_data(slug: str):
    info = DEMO_SIM_BRANDS.get(slug)
    if not info: raise HTTPException(404, "Not found")

    if info["type"] == "campaign":
        cid = info["campaign_id"]
        camp = db.campaigns.find_one({"_id": ObjectId(cid)})
        sims = list(db.simulations.find({"campaign_id": cid, "status": "completed", "results": {"$exists": True}}).sort("created_at", -1))

        # Get videos with signed URLs
        videos = {}
        for v in db.videos.find({"campaign_id": cid}):
            vid = str(v["_id"])
            url = v.get("url", "")
            if v.get("gs_uri"): url = _sign(v["gs_uri"])
            videos[vid] = {"id": vid, "title": v.get("title", "Untitled"), "url": url}

        fmt_sims = []
        for sim in sims[:3]:
            r = sim.get("results", {})
            evals = r.get("evaluations", [])
            if not evals: continue
            vc = {}
            for ev in evals:
                p = ev.get("most_preferred_video", "")
                vc[p] = vc.get(p, 0) + 1
            sv = sorted(vc.items(), key=lambda x: x[1], reverse=True)
            fmt_sims.append({
                "id": str(sim["_id"]), "name": sim.get("name"), "model_provider": sim.get("model_provider"),
                "model_name": sim.get("model_name"), "total_personas": r.get("total_personas_evaluated", 0),
                "total_videos": r.get("total_videos", 0), "evaluations": evals, "video_mapping": r.get("video_mapping", {}),
                "vote_summary": [{"video_number": vn, "votes": c, "percentage": round(c / len(evals) * 100), "video_info": r.get("video_mapping", {}).get(vn, {})} for vn, c in sv],
            })

        return {"brand_name": info["name"], "sim_type": "campaign",
                "campaign": {"name": camp.get("name") if camp else None, "description": camp.get("description") if camp else None, "platform": camp.get("platform") if camp else None},
                "simulations": fmt_sims, "videos": videos, "total_personas": fmt_sims[0]["total_personas"] if fmt_sims else 0, "total_videos": len(videos)}

    # Workflow-based (Stitch Fix)
    wf = db.content_workflows.find_one({"brand_id": info["brand_id"]})
    if not wf: raise HTTPException(404, "No workflow")
    wid = str(wf["_id"])

    sim_node = db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "simulation_testing", "output_data": {"$exists": True, "$ne": {}}})
    pred_node = db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "predictive_modeling", "output_data": {"$exists": True, "$ne": {}}})
    rank_node = db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "content_ranking", "output_data": {"$exists": True, "$ne": {}}})

    # Get stitched videos
    vid_node = db.content_workflow_nodes.find_one({"workflow_id": wid, "stage_key": "video_generation", "output_data": {"$exists": True, "$ne": {}}})
    stitched = []
    if vid_node:
        for v in vid_node["output_data"].get("variations", []):
            if v.get("type") == "stitched":
                url = v.get("video_url", "")
                if v.get("gs_uri"): url = _sign(v["gs_uri"])
                stitched.append({"id": v.get("id"), "model": v.get("model"), "video_url": url, "content_id": v.get("content_id")})

    return {
        "brand_name": info["name"], "sim_type": "workflow", "workflow_title": wf.get("title"),
        "persona_results": (sim_node["output_data"].get("results", []) if sim_node else []),
        "sim_config": (sim_node["output_data"].get("config", {}) if sim_node else {}),
        "predictions": (pred_node["output_data"].get("predictions", []) if pred_node else []),
        "benchmarks": (pred_node["output_data"].get("benchmarks", {}) if pred_node else {}),
        "rankings": (rank_node["output_data"].get("rankings", []) if rank_node else []),
        "rank_weights": (rank_node["output_data"].get("weights", {}) if rank_node else {}),
        "stitched_videos": stitched,
    }
