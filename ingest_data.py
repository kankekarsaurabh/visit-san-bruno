# ingest_data.py

import os
import pandas as pd
import weaviate
import uuid
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from weaviate.auth import AuthApiKey
# ✅ Import the class for bulk import objects
from weaviate.classes.data import DataObject

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CSV_PATH = "Final_Dataset.csv"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CLASS_NAME = "Venue"

# --- Load Data and Model ---
print("📂 Loading data and model...")
df = pd.read_csv(CSV_PATH)
df.dropna(subset=["latitude", "longitude"], inplace=True)
df.reset_index(drop=True, inplace=True)
df.fillna("", inplace=True)

df["category_list"] = df["category"].apply(lambda x: [tag.strip() for tag in str(x).split(',') if tag.strip()])
df["description"] = df["venue_name"] + " - " + df["category"] + " - " + df["address"]
model = SentenceTransformer(MODEL_NAME)

# --- Generate Embeddings ---
print(f"🔄 Generating embeddings for {len(df)} descriptions...")
vectors = model.encode(df["description"].tolist(), show_progress_bar=True)

# --- Connect to Weaviate Cloud Service ---
print("🔌 Connecting to Weaviate Cloud Service...")
try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY)
    )
    client.is_ready()

    # --- Prepare Data for Bulk Import ---
    print("📋 Preparing objects for bulk import...")
    venues_collection = client.collections.get(CLASS_NAME)
    data_objects = []

    for i, row in df.iterrows():
        properties = {
            "venue_name": row.get("venue_name"),
            "category": row.get("category_list"),
            "address": row.get("address"),
            "description": row.get("description"),
            "location": {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"])
            }
        }
        
        data_objects.append(
            DataObject(
                properties=properties,
                vector=vectors[i],
                uuid=uuid.uuid4()
            )
        )

    # --- Execute Bulk Import ---
    print(f"🚀 Starting bulk import of {len(data_objects)} objects...")
    # ✅ REVISED: Use the HTTP-based insert_many method
    result = venues_collection.data.insert_many(data_objects)
    
    if result.has_errors:
        print("❌ Errors occurred during import:")
        for error in result.errors:
            print(error)
    else:
        print("✅ Bulk import complete.")

except Exception as e:
    print(f"❌ An error occurred: {e}")
finally:
    if 'client' in locals() and client.is_connected():
        client.close()
        print("🔌 Connection closed.")