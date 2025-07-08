import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from math import radians, cos, sin, asin, sqrt
import streamlit as st
import folium
from streamlit_folium import st_folium

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv("bay_area_venues.csv")
df['category'] = df['category'].str.lower()
df['tags_list'] = df['category'].apply(lambda x: [x])
df['all_features'] = df['tags_list']
df['description'] = df['category'] + " venue in " + df['address']

# ------------------------
# SBERT model
# ------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
df['embedding'] = df['description'].apply(lambda x: model.encode(x, convert_to_tensor=True))

# ------------------------
# Haversine distance
# ------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))  # in KM

# ------------------------
# Get Recommendations
# ------------------------
def get_recommendations(prompt, user_lat, user_lon, radius_km=5, top_n=5):
    user_embedding = model.encode(prompt, convert_to_tensor=True)
    similarities = [float(util.pytorch_cos_sim(user_embedding, emb)[0]) for emb in df['embedding']]

    df['similarity'] = similarities
    df['distance_km'] = df.apply(lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']), axis=1)
    filtered = df[df['distance_km'] <= radius_km].sort_values(by='similarity', ascending=False).head(top_n)

    return filtered

# ------------------------
# Streamlit App
# ------------------------
st.title("🗺️ Visit San Bruno - Curated Experience Recommender")

with st.sidebar:
    st.markdown("### 👤 Enter Your Experience Prompt")
    prompt = st.text_input("What do you want to do?", "sushi and karaoke then wine")

    st.markdown("### 📍 Location (defaults to San Bruno)")
    lat = st.number_input("Latitude", value=37.63, format="%.5f")
    lon = st.number_input("Longitude", value=-122.42, format="%.5f")
    radius = st.slider("Search Radius (km)", min_value=1, max_value=20, value=5)

if prompt:
    st.write("✅ **Matched Prompt:**", prompt)
    results = get_recommendations(prompt, lat, lon, radius_km=radius)

    if not results.empty:
        st.success("🎉 Found {} places within {} km".format(len(results), radius))
        st.dataframe(results[['venue_name', 'category', 'address', 'distance_km']].round(2))

        # 🗺️ Show on Map
        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Marker([lat, lon], tooltip="You are here", icon=folium.Icon(color='blue')).add_to(m)

        for _, row in results.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                tooltip=f"{row['venue_name']} ({row['category']})",
                popup=row['address'],
                icon=folium.Icon(color='green')
            ).add_to(m)

        st_folium(m, width=700, height=500)
    else:
        st.warning("No matching venues found. Try another prompt or increase the radius.")
