import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Boston Crime Hotspots", layout="wide")

st.title("Boston Crime Hotspots (2023–Present)")
st.caption("Hexbin hotspot visualization with monthly playback and filtering.")

@st.cache_data
def load_hex_monthly(path: str):
    # GeoParquet via GeoPandas
    gdf = gpd.read_parquet(path)
    # Ensure date is datetime
    gdf["date"] = pd.to_datetime(gdf["date"])
    # Reproject to WGS84 for web mapping
    gdf = gdf.to_crs(epsg=4326)
    return gdf

# ---- Load data ----
DATA_PATH = "data/boston_crime_hex_monthly.parquet"
hex_monthly = load_hex_monthly(DATA_PATH)

# ---- Sidebar controls ----
st.sidebar.header("Controls")

# Date selector
available_dates = sorted(hex_monthly["date"].dropna().unique())
selected_date = st.sidebar.selectbox("Month", available_dates, index=len(available_dates)-1)

# Crime group filter (if present)
if "crime_group" in hex_monthly.columns:
    groups = ["All"] + sorted(hex_monthly["crime_group"].dropna().unique().tolist())
    selected_group = st.sidebar.selectbox("Crime Group", groups, index=0)
else:
    selected_group = "All"
    st.sidebar.info("No crime_group column in hex_monthly yet. (We can add it next.)")

# Hotspot toggle
show_hotspots_only = st.sidebar.checkbox("Show hotspots only (top 5% each month)", value=True)

# Playback
play = st.sidebar.checkbox("Play animation", value=False)
fps = st.sidebar.slider("Speed (frames/sec)", 1, 10, 3)

# ---- Filter data ----
df = hex_monthly[hex_monthly["date"] == selected_date].copy()

if selected_group != "All" and "crime_group" in df.columns:
    df = df[df["crime_group"] == selected_group]

if show_hotspots_only and "is_hotspot" in df.columns:
    df = df[df["is_hotspot"] == True]

# If empty, warn and stop
if df.empty:
    st.warning("No data for this selection. Try a different month/filter.")
    st.stop()

# ---- Convert polygons to coordinates for PyDeck ----
# PyDeck PolygonLayer expects coordinates in [lon, lat] format
def geom_to_coords(geom):
    # Works for Polygon; if MultiPolygon we can handle later
    if geom.geom_type == "Polygon":
        return [list(geom.exterior.coords)]
    elif geom.geom_type == "MultiPolygon":
        # Take all polygon exteriors
        return [list(p.exterior.coords) for p in geom.geoms]
    return None

df["coordinates"] = df["geometry"].apply(geom_to_coords)
df = df.dropna(subset=["coordinates"])

# ---- KPIs ----
col1, col2, col3 = st.columns(3)
col1.metric("Hexes Shown", f"{len(df):,}")
col2.metric("Incidents (sum)", f"{int(df['incident_count'].sum()):,}")
if "is_hotspot" in df.columns:
    col3.metric("Hotspot Hexes", f"{int(df['is_hotspot'].sum()):,}")
else:
    col3.metric("Hotspot Hexes", "N/A")

# ---- Map ----
view_state = pdk.ViewState(
    latitude=42.33,
    longitude=-71.06,
    zoom=10.5,
    pitch=40,
)

layer = pdk.Layer(
    "PolygonLayer",
    data=df,
    get_polygon="coordinates",
    get_elevation="incident_count",
    elevation_scale=8,
    extruded=True,
    wireframe=False,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/light-v9",
    tooltip={"text": "Incidents: {incident_count}"},
)

st.pydeck_chart(deck, use_container_width=True)

# ---- Playback (simple) ----
if play:
    import time
    placeholder = st.empty()

    for d in available_dates:
        df2 = hex_monthly[hex_monthly["date"] == d].copy()

        if selected_group != "All" and "crime_group" in df2.columns:
            df2 = df2[df2["crime_group"] == selected_group]

        if show_hotspots_only and "is_hotspot" in df2.columns:
            df2 = df2[df2["is_hotspot"] == True]

        df2["coordinates"] = df2["geometry"].apply(geom_to_coords)
        df2 = df2.dropna(subset=["coordinates"])

        layer2 = pdk.Layer(
            "PolygonLayer",
            data=df2,
            get_polygon="coordinates",
            get_elevation="incident_count",
            elevation_scale=8,
            extruded=True,
            wireframe=False,
        )

        deck2 = pdk.Deck(
            layers=[layer2],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v9",
            tooltip={"text": f"{d.strftime('%Y-%m')} | Incidents: {{incident_count}}"},
        )

        placeholder.pydeck_chart(deck2, use_container_width=True)
        time.sleep(1 / fps)
