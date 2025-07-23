# sanbruno.py

import json
import os
import traceback
from math import asin, cos, radians, sin, sqrt

import pandas as pd
import requests
import torch
import weaviate
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
from tinydb import Query, TinyDB
from weaviate.auth import AuthApiKey

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
class Config:
    """Configuration class for easy management of settings."""
    VENUE_DATA_PATH = "Final_Dataset.csv"
    CHAT_HISTORY_DB_PATH = "chat_history.json"
    NEARBY_RADIUS_MILES = 3.1 
    
    # Model & API Settings
    SBERT_MODEL = "BAAI/bge-base-en-v1.5"
    LLM_MODEL = "llama3-8b-8192" 
    
    # Cloud Service Credentials from Environment
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    DEFAULT_LAT = 37.6283
    DEFAULT_LON = -122.4166
    SAN_BRUNO_BOUNDS = { "north": 37.65, "south": 37.60, "west": -122.46, "east": -122.40 }

# --- Application Setup ---
app = FastAPI(title="Visit San Bruno API")
cfg = Config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Resources ---
try:
    if not all([cfg.WEAVIATE_URL, cfg.WEAVIATE_API_KEY, cfg.GROQ_API_KEY]):
        raise ValueError("Missing required environment variables.")

    print("🔄 Loading SBERT and data...")
    sbert_model = SentenceTransformer(cfg.SBERT_MODEL)
    df = pd.read_csv(cfg.VENUE_DATA_PATH)
    
    df.fillna({"category": ""}, inplace=True)
    df['category_list'] = df['category'].apply(lambda x: [tag.strip() for tag in str(x).split(',') if tag.strip()])
    
    unique_categories = df.explode('category_list')['category_list'].dropna().unique().tolist()
    
    category_embeddings = sbert_model.encode(unique_categories, convert_to_tensor=True)
    print(f"📚 Found {len(unique_categories)} unique category tags.")
    
    print("🔌 Initializing Weaviate client...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=cfg.WEAVIATE_URL,
        auth_credentials=AuthApiKey(api_key=cfg.WEAVIATE_API_KEY)
    )
    client.is_ready()

except Exception as e:
    raise RuntimeError(f"FATAL: Failed to initialize resources: {e}")

chat_history_db = TinyDB(cfg.CHAT_HISTORY_DB_PATH)

# --- Pydantic Models ---
class ChatInput(BaseModel):
    session_id: str
    message: str
    lat: float | None = None
    lon: float | None = None

# --- Helper Functions ---
def _is_in_san_bruno(lat: float, lon: float) -> bool:
    bounds = cfg.SAN_BRUNO_BOUNDS
    return (bounds["south"] <= lat <= bounds["north"]) and (bounds["west"] <= lon <= bounds["east"])

def _haversine_in_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 3956 * 2 * asin(sqrt(a))

def _split_steps(prompt: str):
    connectors = [" then ", " followed by ", " after ", "→", ",", " and "]
    lower_prompt = prompt.lower()
    for conn in connectors:
        if conn in lower_prompt:
            return [p.strip() for p in prompt.split(conn) if p.strip()]
    return [prompt.strip()]

def _recommend_venues(prompt: str, lat: float, lon: float):
    # (This function remains unchanged from the previous version)
    steps = _split_steps(prompt)
    recommendations = []
    for step in steps:
        query_embedding = sbert_model.encode(step).tolist()
        try:
            venues_collection = client.collections.get("Venue")
            response = venues_collection.query.near_vector(
                near_vector=query_embedding,
                limit=10,
                return_properties=["venue_name", "category", "address", "location"]
            )
            venues = [obj.properties for obj in response.objects]
        except Exception as e:
            print(f"❌ WEAVIATE QUERY FAILED for step '{step}': {e}")
            venues = []

        venues_df = pd.DataFrame(venues)
        nearby_venues = pd.DataFrame()
        if not venues_df.empty and "location" in venues_df.columns:
            venues_df["latitude"] = venues_df["location"].apply(lambda loc: loc.latitude if loc else None)
            venues_df["longitude"] = venues_df["location"].apply(lambda loc: loc.longitude if loc else None)
            venues_df.dropna(subset=["latitude", "longitude"], inplace=True)
            venues_df["distance_miles"] = venues_df.apply(lambda r: _haversine_in_miles(lat, lon, r["latitude"], r["longitude"]), axis=1)
            nearby_venues = venues_df[venues_df["distance_miles"] <= cfg.NEARBY_RADIUS_MILES]

        if nearby_venues.empty and not df.empty:
            query_tensor = torch.tensor(query_embedding).to(category_embeddings.device)
            category_scores = util.cos_sim(query_tensor, category_embeddings)[0]
            best_cat_idx = torch.argmax(category_scores).item()
            best_category = unique_categories[best_cat_idx]
            fallback_df = df[df['category_list'].apply(lambda tags: best_category in tags)].copy()
            if not fallback_df.empty and "latitude" in fallback_df.columns and "longitude" in fallback_df.columns:
                fallback_df["distance_miles"] = fallback_df.apply(lambda r: _haversine_in_miles(lat, lon, r.get("latitude", 0), r.get("longitude", 0)), axis=1)
                nearby_venues = fallback_df.sort_values(by="distance_miles").head(3)
        
        nearby_venues = nearby_venues.fillna("")
        recommendations.append({ "query": step, "venues": nearby_venues.to_dict(orient="records") })
    return recommendations

def _is_detail_query(message: str, plan: list) -> bool:
    # (This function remains unchanged)
    if not plan: return False
    prompt = f'A user has this travel plan: {json.dumps(plan)}. Their latest message is: "{message}". Is this message asking for details about the *existing* plan? Answer ONLY with "YES" or "NO".'
    try:
        headers = {"Authorization": f"Bearer {cfg.GROQ_API_KEY}"}
        response = requests.post(
            cfg.GROQ_API_URL, headers=headers,
            json={"model": cfg.LLM_MODEL, "messages": [{"role": "system", "content": prompt}], "temperature": 0.0},
            timeout=10
        )
        response.raise_for_status()
        reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "NO").strip().upper()
        print(f"🧠 Detail Query Check: {reply}")
        return "YES" in reply
    except Exception as e:
        print(f"❌ Detail Query Check failed: {e}")
        return False

# ✅ NEW HELPER FUNCTION
def _is_query_too_broad(message: str) -> bool:
    """Uses an LLM to check if a user's query is too vague for recommendations."""
    prompt = f"""
    You are an AI assistant that determines if a user's request for a recommendation is too broad or specific enough.
    A specific request mentions a type of place, food, or activity (e.g., "sushi," "a park with a playground," "a quiet cafe").
    A broad request is a general statement of need (e.g., "I'm hungry," "I'm bored," "what should I do?").

    Analyze the user's message: "{message}"

    Is this request too broad to make a specific recommendation? Answer ONLY with "YES" or "NO".
    """
    try:
        headers = {"Authorization": f"Bearer {cfg.GROQ_API_KEY}"}
        response = requests.post(
            cfg.GROQ_API_URL, headers=headers,
            json={"model": cfg.LLM_MODEL, "messages": [{"role": "system", "content": prompt}], "temperature": 0.0},
            timeout=10
        )
        response.raise_for_status()
        reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "NO").strip().upper()
        print(f"🧠 Broad Query Check: {reply}")
        return "YES" in reply
    except Exception as e:
        print(f"❌ Broad Query Check failed: {e}")
        return False

# --- API Endpoints ---
@app.post("/chat", summary="Main stateful conversational endpoint")
def chat(input: ChatInput):
    Session = Query()
    session_data = chat_history_db.get(Session.session_id == input.session_id)
    
    if not session_data:
        session_data = { "session_id": input.session_id, "messages": [{"role": "system", "content": "You are a friendly and helpful travel assistant for San Bruno."}], "plan": [] }
    
    session_data["messages"].append({"role": "user", "content": input.message})
    current_plan = session_data.get("plan", [])
    
    target_lat, target_lon = cfg.DEFAULT_LAT, cfg.DEFAULT_LON
    if input.lat is not None and input.lon is not None:
        if _is_in_san_bruno(input.lat, input.lon):
            target_lat, target_lon = input.lat, input.lon
            print(f"📍 Using user's location in San Bruno: ({target_lat}, {target_lon})")
        else:
            print(f"📍 User location is outside San Bruno. Using default location.")
    else:
        print("📍 No user location provided. Using default location.")

    context_for_llm = ""
    
    # ✅ REVISED LOGIC: Add a check for broad queries
    if _is_detail_query(input.message, current_plan):
        print("➡️ Handling: DETAIL query")
        context_for_llm = "Use the following data from the current plan to answer the user's question. The distances are in miles.\n" + json.dumps(current_plan, indent=2)
    
    elif _is_query_too_broad(input.message):
        print("➡️ Handling: BROAD query")
        context_for_llm = "You are a helpful assistant. The user's request is too vague to make a recommendation. Your task is to ask a clarifying question to help them narrow down their options. For example, if they say 'I'm hungry,' ask 'What kind of food are you in the mood for?'. If they say 'I'm bored,' ask 'Are you looking for an indoor or outdoor activity?'"

    else:
        print("➡️ Handling: CREATE/UPDATE query")
        recommendations = _recommend_venues(input.message, target_lat, target_lon)
        
        found_venues = any(rec.get("venues") for rec in recommendations)
        if found_venues:
            new_plan = []
            for rec in recommendations:
                if rec["venues"]:
                    top_venue = rec["venues"][0]
                    new_plan.append({
                        "step_query": rec["query"], "venue_name": top_venue.get("venue_name"),
                        "category": top_venue.get("category"), 
                        "distance_miles": round(top_venue.get("distance_miles", 0), 2)
                    })
            session_data["plan"] = new_plan
            context_for_llm = (
                "You have just created a new itinerary. Summarize it for the user in a friendly, step-by-step list."
                "Mention the venue name for each step and its distance in miles.\n"
                "IMPORTANT: Do NOT use any Markdown formatting like asterisks for bolding. Return only plain text.\n"
                + json.dumps(new_plan, indent=2)
            )
        else:
            print("➡️ Handling: No Venues Found")
            context_for_llm = f"""
            Your ONLY task is to inform the user that their search returned no results.
            The user's original failed request was: '{input.message}'.
            Respond ONLY with a message similar to this template: "I'm sorry, I couldn't find any local venues matching '[user's request]' in my database. Please try a different search."
            """

    # --- Generate Final LLM Response ---
    try:
        messages_for_llm = list(session_data["messages"])
        if context_for_llm:
            messages_for_llm.append({"role": "system", "content": context_for_llm})
        
        headers = {"Authorization": f"Bearer {cfg.GROQ_API_KEY}"}
        response = requests.post(
            cfg.GROQ_API_URL, headers=headers,
            json={"model": cfg.LLM_MODEL, "messages": messages_for_llm, "temperature": 0.7}, # Temperature can be higher for conversational replies
            timeout=60
        )
        response.raise_for_status()
        reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="The request to the AI model timed out.")
    except Exception as e:
        print(f"❌ Final LLM call failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to get a response from the assistant.")
        
    session_data["messages"].append({"role": "assistant", "content": reply})
    chat_history_db.upsert(session_data, Session.session_id == input.session_id)
    
    return {"reply": reply, "session_id": input.session_id}