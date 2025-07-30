import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from dem_handler import load_dem
from aop import crop_dem, latlon_to_pixel_bounds
from terrain_analysis import calculate_slope, mask_invalid_terrain
from radar_simulation import get_radar_coverage
from radar_optimization import RadarOptimizer
from user_input import compute_radar_range

st.set_page_config(page_title="Optimal Radar Deployment", layout="centered")

st.title("📡 Optimal Radar Deployment Web App")

# --- Model Selection ---
use_existing = st.radio("Do you want to use an existing saved model?",
                        ["No, Train New", "Yes, Use Saved Model"])

models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(models_dir, exist_ok=True)
saved_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

selected_model = None
if use_existing == "Yes, Use Saved Model":
    if saved_models:
        model_choice = st.selectbox("Select a saved model", saved_models)
        selected_model = os.path.join(models_dir, model_choice)
    else:
        st.warning("No saved models found, please train a new one.")

# --- Inputs ---
area = st.text_input("Enter the state name (e.g., rajasthan):", "rajasthan")
num_radars = st.number_input("Enter number of radars to deploy:", min_value=1, value=1, step=1)
bbox = st.text_input("Enter bounding box (min_lat,min_lon,max_lat,max_lon)",
                     "25.0,70.0,26.0,71.0")

# --- Dynamic Radar Inputs ---
st.subheader("Radar Specifications")
radars = []
for i in range(num_radars):
    st.markdown(f"#### Radar {i+1}")
    Pt = st.number_input(f"Transmit Power (Pt) for Radar {i+1}", value=1000.0, key=f"Pt_{i}")
    G_dBi = st.number_input(f"Antenna Gain (dBi) for Radar {i+1}", value=30.0, key=f"G_{i}")
    Frequency_Hz = st.number_input(f"Frequency (Hz) for Radar {i+1}", value=3e9, key=f"F_{i}")
    Height = st.number_input(f"Radar Height (m) for Radar {i+1}", value=50.0, key=f"H_{i}")
    radars.append({
        "Pt": Pt,
        "G_dBi": G_dBi,
        "Frequency_Hz": Frequency_Hz,
        "Height": Height
    })

# --- Run Button ---
if st.button("Run Radar Optimization"):
    # --- Load DEM ---
    try:
        dem, transform, _ = load_dem(area)
    except FileNotFoundError:
        st.error(f"DEM file not found for '{area}'. Ensure it's in Dataset/{area}/")
        st.stop()

    # --- Crop DEM ---
    coords = list(map(float, bbox.split(',')))
    x_min, y_min, x_max, y_max = latlon_to_pixel_bounds(transform, coords)
    cropped_dem = crop_dem(dem, x_min, y_min, x_max, y_max)

    # Plot DEM
    fig1, ax1 = plt.subplots()
    ax1.imshow(cropped_dem, cmap="terrain")
    ax1.set_title("Cropped AOP DEM")
    st.pyplot(fig1)

    # --- Terrain Analysis ---
    slope_map = calculate_slope(cropped_dem)
    invalid_mask, roughness_map, _ = mask_invalid_terrain(cropped_dem, slope_map, slope_thresh=30)

    # Roughness Plot
    fig2, ax2 = plt.subplots()
    ax2.imshow(roughness_map, cmap="viridis")
    ax2.set_title("Terrain Roughness")
    st.pyplot(fig2)

    # Slope > threshold Plot
    fig3, ax3 = plt.subplots()
    ax3.imshow(slope_map > np.tan(np.radians(30)), cmap="Reds")
    ax3.set_title("Slopes > 30°")
    st.pyplot(fig3)

    # Invalid terrain mask
    fig4, ax4 = plt.subplots()
    ax4.imshow(invalid_mask, cmap="gray")
    ax4.set_title("Combined Invalid Terrain Mask")
    st.pyplot(fig4)

    pixel_size = 30  # meters
    if selected_model:
        # Load saved positions
        with open(selected_model, 'rb') as f:
            optimized_positions = pickle.load(f)
        st.success(f"Loaded saved model: {selected_model}")
    else:
        # --- Optimization ---
        optimizer = RadarOptimizer(
            dem=cropped_dem,
            invalid_mask=invalid_mask,
            num_radars=num_radars,
            radius_pixels=100,  # placeholder, updated per radar
            radar_height=50
        )
        optimized_positions = optimizer.optimize(generations=5, sol_per_pop=5)

        # Save Model
        import datetime
        filename = f"radar_positions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = os.path.join(models_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(optimized_positions, f)
        st.success(f"Optimized radar positions saved: {filename}")

    # --- Coverage ---
    final_coverage = np.zeros_like(cropped_dem, dtype=bool)
    fig5, ax5 = plt.subplots()
    ax5.imshow(cropped_dem, cmap="terrain")

    for i, (y, x) in enumerate(optimized_positions):
        radar = radars[i % len(radars)]
        r_max = compute_radar_range(radar["Pt"], radar["G_dBi"], radar["Frequency_Hz"])
        radius_pixels = int(r_max / pixel_size)
        coverage = get_radar_coverage(cropped_dem, (x, y), radius_pixels, radar["Height"])
        coverage &= ~invalid_mask
        final_coverage |= coverage
        ax5.imshow(coverage, cmap="Greens", alpha=0.3)
        ax5.scatter(x, y, c="red", marker="x", s=100)
        ax5.text(x, y, f"R{i+1}", color="white", ha="center", va="center")

    ax5.set_title("Optimized Radar Coverage Map")
    st.pyplot(fig5)


    # --- Coverage Stats ---
    pixel_area_km2 = (pixel_size / 1000) ** 2
    total_area_km2 = cropped_dem.size * pixel_area_km2
    covered_area_km2 = np.sum(final_coverage) * pixel_area_km2
    coverage_percent = (np.sum(final_coverage) / cropped_dem.size) * 100
    st.markdown(f"**AOP Total Area**: {total_area_km2:.2f} km²")
    st.markdown(f"**Optimized Covered Area**: {covered_area_km2:.2f} km²")
    st.markdown(f"**Coverage Percentage**: {coverage_percent:.2f}%")

    # --- Display radar coordinates at the end ---
    st.subheader("📍 Optimized Radar Coordinates")
    for i, (y, x) in enumerate(optimized_positions):
        st.write(f"Radar {i + 1}: (Row: {y}, Col: {x})")

    # coordinates in lat/long form
    from rasterio.transform import Affine
    # --- Display radar coordinates at the end ---
    st.subheader("📍 Optimized Radar Coordinates (Pixel & Lat/Lon)")

    for i, (y, x) in enumerate(optimized_positions):
        lon, lat = transform * (x, y)
        st.write(f"Radar {i + 1}: Pixel(Row={y}, Col={x}) → Lat: {lat:.6f}, Lon: {lon:.6f}")















