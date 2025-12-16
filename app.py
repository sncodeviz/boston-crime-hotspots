import os
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Boston Crime Hotspots", layout="wide")

st.title("Boston Crime Hotspots (2023–Present)")
st.caption("Interactive hexbin hotspot explorer with monthly playback and optional persistence view.")

DATA_DIR = "data"
MONTHLY_FILE = os.path.join(DATA_DIR, "boston_crime_hex_monthly.parquet")
STABILITY_FILE = os.path.join(DATA_DIR, "boston_crime_hex_stability.parquet")


# ---------- Helpers ----------
@st.cache_data
def load_geoparquet(path: str) -> gpd.GeoDataFrame:
    """Load a GeoParquet file and ensure WGS84 (lat/lon) for web mapping."""
    gdf = gpd.read_parquet(path)
    if "date" in gdf.columns:
        gdf["date"] = pd.to_datetime(gdf["date"], errors="coerce")

    # Reproject to EPSG:4326 for folium
    if gdf.crs is not None and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
        gdf = gdf.to_crs(epsg=4326)
    elif gdf.crs is None:
        # If CRS missing, assume it is already WGS84 for mapping (best-effort)
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    return gdf


def safe_quantile(series: pd.Series, q: float, default: float = 0.0) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.quantile(q)) if len(s) else default


def color_scale(value: float, vmin: float, vmax: float) -> str:
    """Simple single-hue ramp (darker = higher)."""
    if vmax <= vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0, 1))

    r = int(255 * (0.35 + 0.65 * t))
    g = int(255 * (0.90 - 0.75 * t))
    b = int(255 * (0.90 - 0.75 * t))
    return f"#{r:02x}{g:02x}{b:02x}"


def _json_safe_series(s: pd.Series) -> pd.Series:
    """Convert a Series to JSON-safe Python primitives."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m-%d")

    # Convert numpy scalars to Python scalars
    def conv(x):
        if pd.isna(x):
            return None
        if isinstance(x, np.generic):
            return x.item()
        return x

    return s.apply(conv)


def make_map(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    tooltip_cols: list[str],
    center=(42.33, -71.06),
    zoom=11
) -> folium.Map:
    """
    Create a Folium map with polygon styling based on value_col.
    Uses a JSON-safe GeoJSON conversion to avoid serialization errors.
    """
    # Keep only what we need to avoid non-serializable columns
    keep_cols = ["geometry"] + sorted(set([value_col] + tooltip_cols))
    g = gdf[keep_cols].copy()

    # Ensure WGS84
    if g.crs is not None and str(g.crs).lower() not in ("epsg:4326", "wgs84"):
        g = g.to_crs(epsg=4326)
    elif g.crs is None:
        g = g.set_crs(epsg=4326, allow_override=True)

    # Make columns JSON-safe
    for c in keep_cols:
        if c == "geometry":
            continue
        g[c] = _json_safe_series(g[c])

    # Value range for styling
    vals = pd.to_numeric(g[value_col], errors="coerce").fillna(0)
    vmin, vmax = float(vals.min()), float(vals.max())

    def style_fn(feature):
        val = feature["properties"].get(value_col, 0) or 0
        try:
            val = float(val)
        except Exception:
            val = 0.0
        return {
            "fillColor": color_scale(val, vmin, vmax),
            "color": "#333333",
            "weight": 0.3,
            "fillOpacity": 0.65,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=tooltip_cols,
        aliases=[f"{c}: " for c in tooltip_cols],
        localize=True,
        sticky=False
    )

    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)

    # Convert to GeoJSON dict safely
    geojson_dict = json.loads(g.to_json())

    folium.GeoJson(
        geojson_dict,
        style_function=style_fn,
        tooltip=tooltip,
        name="Hotspots"
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ---------- Load data ----------
if not os.path.exists(MONTHLY_FILE):
    st.error(f"Missing required file: `{MONTHLY_FILE}`. Add it under /data/ in your repo.")
    st.stop()

hex_monthly = load_geoparquet(MONTHLY_FILE)

stability_available = os.path.exists(STABILITY_FILE)
hex_stability = load_geoparquet(STABILITY_FILE) if stability_available else None

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")

modes = ["Monthly hotspots (play)"]
if stability_available:
    modes.append("Persistent hotspots")
mode = st.sidebar.radio("Mode", modes, index=0)

show_hotspots_only = st.sidebar.checkbox(
    "Show hotspots only (top 5% each month)",
    value=True,
    help="Uses `is_hotspot` if present; otherwise computes a top-5% cutoff per month."
)

# Playback controls (monthly mode)
play = False
fps = 3
if mode == "Monthly hotspots (play)":
    play = st.sidebar.checkbox("Play animation", value=False)
    fps = st.sidebar.slider("Speed (frames/sec)", 1, 10, 3)

# Crime-group filtering (only if present)
crime_group_col = "crime_group" if "crime_group" in hex_monthly.columns else None
selected_group = "All"
if crime_group_col:
    groups = ["All"] + sorted([g for g in hex_monthly[crime_group_col].dropna().unique().tolist()])
    selected_group = st.sidebar.selectbox("Crime Group", groups, index=0)
else:
    st.sidebar.info("Crime-group filtering is off (no `crime_group` column in monthly file).")

# ---------- Monthly mode ----------
if mode == "Monthly hotspots (play)":
    if "date" not in hex_monthly.columns:
        st.error("Monthly file must include a `date` column (Phase 3.4 output).")
        st.stop()

    available_dates = sorted(hex_monthly["date"].dropna().unique())
    if not available_dates:
        st.error("No valid dates found in the monthly dataset.")
        st.stop()

    selected_date = st.sidebar.selectbox("Month", available_dates, index=len(available_dates) - 1)

    def filter_month(d: pd.Timestamp) -> gpd.GeoDataFrame:
        df = hex_monthly[hex_monthly["date"] == d].copy()

        if selected_group != "All" and crime_group_col:
            df = df[df[crime_group_col] == selected_group].copy()

        if "incident_count" not in df.columns:
            st.error("Monthly dataset must include `incident_count`.")
            st.stop()

        # Ensure numeric for styling
        df["incident_count"] = pd.to_numeric(df["incident_count"], errors="coerce").fillna(0)

        # Hotspots-only filtering
        if show_hotspots_only:
            if "is_hotspot" in df.columns:
                df = df[df["is_hotspot"] == True].copy()
            else:
                cutoff = safe_quantile(df["incident_count"], 0.95, default=0.0)
                df = df[df["incident_count"] >= cutoff].copy()

        df = df[df.geometry.notna()].copy()
        return df

    df_frame = filter_month(selected_date)

    if df_frame.empty:
        st.warning("No data for this month/filter. Try turning off 'hotspots only' or choosing another month.")
        st.stop()

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Hexes shown", f"{len(df_frame):,}")
    k2.metric("Incidents (sum)", f"{int(df_frame['incident_count'].sum()):,}")
    if "is_hotspot" in df_frame.columns:
        k3.metric("Hotspot hexes", f"{int(df_frame['is_hotspot'].sum()):,}")
    else:
        k3.metric("Hotspot hexes", "Computed")

    # Tooltip fields
    tooltip_cols = ["incident_count"]
    if crime_group_col and selected_group == "All":
        tooltip_cols = [crime_group_col] + tooltip_cols

    st.subheader(f"Map: {pd.to_datetime(selected_date).strftime('%Y-%m')}")
    m = make_map(df_frame, value_col="incident_count", tooltip_cols=tooltip_cols)
    st_folium(m, width=1200, height=650)

    # Playback
    if play:
        st.subheader("Playback")
        map_placeholder = st.empty()
        label_placeholder = st.empty()

        for d in available_dates:
            df_anim = filter_month(d)
            if df_anim.empty:
                continue
            label_placeholder.markdown(f"**{pd.to_datetime(d).strftime('%Y-%m')}**")
            m2 = make_map(df_anim, value_col="incident_count", tooltip_cols=tooltip_cols)
            map_placeholder = st_folium(m2, width=1200, height=650)
            time.sleep(1 / max(1, fps))

# ---------- Persistent mode ----------
else:
    if hex_stability is None or hex_stability.empty:
        st.error("Stability dataset could not be loaded.")
        st.stop()

    if "hotspot_frequency" not in hex_stability.columns:
        st.error("Stability file must include `hotspot_frequency` (Phase 3.4 output).")
        st.stop()

    min_freq = st.sidebar.slider(
        "Min hotspot frequency",
        0.0, 1.0, 0.30, 0.05,
        help="Share of months a hex is classified as a hotspot. Higher = more persistent."
    )

    df_persist = hex_stability.copy()
    df_persist["hotspot_frequency"] = pd.to_numeric(df_persist["hotspot_frequency"], errors="coerce").fillna(0)
    df_persist = df_persist[df_persist["hotspot_frequency"] >= min_freq].copy()
    df_persist = df_persist[df_persist.geometry.notna()].copy()

    if df_persist.empty:
        st.warning("No hexes meet that frequency threshold. Lower the slider.")
        st.stop()

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Persistent hotspot hexes", f"{len(df_persist):,}")
    k2.metric("Min frequency", f"{min_freq:.2f}")
    k3.metric("Max frequency", f"{float(df_persist['hotspot_frequency'].max()):.2f}")

    # Use a percent column for map styling
    df_persist["frequency_pct"] = (df_persist["hotspot_frequency"] * 100).round(1)

    st.subheader("Map: Persistent hotspots (stability)")
    m = make_map(df_persist, value_col="frequency_pct", tooltip_cols=["frequency_pct"])
    st_folium(m, width=1200, height=650)

st.markdown("---")
st.caption(
    "Notes: This visualization uses aggregated hex bins (not addresses). "
    "Incident reports reflect reporting/recording practices and are not a direct measure of true crime prevalence."
)
