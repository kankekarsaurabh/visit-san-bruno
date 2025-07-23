# create_weaviate_schema.py
import weaviate
import os
from dotenv import load_dotenv
from weaviate.auth import AuthApiKey
# ✅ Import new v4 classes for configuration
from weaviate.classes.config import Configure, Property, DataType

load_dotenv()

# --- Configuration ---
CLASS_NAME = "Venue"

# --- Connect to Weaviate ---
try:
    print("🔌 Connecting to Weaviate Cloud Service...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
    )
    client.is_ready()

    # --- Delete existing collection if it exists ---
    if client.collections.exists(CLASS_NAME):
        print(f"Collection '{CLASS_NAME}' already exists. Deleting it...")
        client.collections.delete(CLASS_NAME)
        print(f"Collection '{CLASS_NAME}' deleted.")

    # --- Create New Collection with v4 Syntax ---
    print(f"Creating new collection: '{CLASS_NAME}'...")
    venues = client.collections.create(
        name=CLASS_NAME,
        description="Venues in the San Francisco Bay Area",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="venue_name", data_type=DataType.TEXT),
            # Use DataType.TEXT_ARRAY for a list of strings
            Property(name="category", data_type=DataType.TEXT_ARRAY),
            Property(name="address", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="location", data_type=DataType.GEO_COORDINATES),
        ]
    )
    print("✅ Schema created successfully.")

except Exception as e:
    print(f"❌ An error occurred: {e}")
finally:
    # Best practice: always close the connection
    if 'client' in locals() and client.is_connected():
        client.close()
        print("🔌 Connection closed.")