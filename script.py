import streamlit as st
import google.generativeai as genai
import re
import json
from geopy.geocoders import Nominatim
from datetime import date
from sentinelhub import (
    SHConfig, BBox, CRS, DataCollection, SentinelHubCatalog, SentinelHubRequest, MimeType, SentinelHubDownloadClient
)
import numpy as np
import matplotlib.pyplot as plt

# API Keys & Config
try:
    gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
    sentinelhub_client_id = st.secrets["api_keys"]["sentinelhub_client_id"]
    sentinelhub_client_secret = st.secrets["api_keys"]["sentinelhub_client_secret"]
except KeyError:
    st.error("❌ API keys are missing in `secrets.toml`. Please configure the file.")
    st.stop()

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

config = SHConfig()
config.sh_client_id = sentinelhub_client_id
config.sh_client_secret = sentinelhub_client_secret

geolocator = Nominatim(user_agent="geo_query")

# Sentinel Product Mapping
SENTINELHUB_PRODUCTS = {
    "Sentinel-1 GRD": "Radar imaging for flood detection, land deformation, and sea ice monitoring.",
    "Sentinel-2 L1C": "Top-of-atmosphere reflectance, useful for land cover and vegetation analysis.",
    "Sentinel-2 L2A": "Surface reflectance corrected for atmospheric effects, ideal for NDVI and agricultural monitoring."
}

product_mapping = {
    "Sentinel-1 GRD": DataCollection.SENTINEL1_IW,
    "Sentinel-2 L1C": DataCollection.SENTINEL2_L1C,
    "Sentinel-2 L2A": DataCollection.SENTINEL2_L2A
}

# Extract Query Details
def extract_query_details(user_query):
    prompt = f"""
    Extract the following information from the user's query:
    - The location (city, country, or coordinates).
    - The time period (specific dates, relative time ranges like "past year", or historical references).
    - The type of analysis requested (e.g., NDVI trend, flood detection, deforestation analysis, urban expansion, etc.).

    Example Query: "What's the NDVI trend in Singapore over the past year?"

    Today's date: {date.today().strftime("%Y-%m-%d")}  # Example: "2025-03-01"

    Expected Response (JSON format):
    {{
        "location": "Singapore",
        "time_range": ["2023-01-01", "2024-01-01"],
        "analysis_type": "NDVI trend"
    }}

    User Query: "{user_query}"
    """
    
    response = model.generate_content(prompt)

    try:
      # Extract JSON from response using regex (handles cases where Gemini returns extra text)
      match = re.search(r"```json\n(.*?)\n```", response.text, re.DOTALL)
      if match:
          json_text = match.group(1)
      else:
          json_text = response.text.strip()

      return json.loads(json_text)
    except json.JSONDecodeError:
        print("Error: Gemini messed up.")
        return {}

# Select Sentinel Product
def select_sentinel_product(analysis_type):
    product_selection_prompt = f"""
    You are selecting the best SentinelHub product for a geospatial analysis task.

    Analysis Type: "{analysis_type}"

    Available SentinelHub Products:
    { {k: v for k, v in SENTINELHUB_PRODUCTS.items()} }

    Select the most appropriate product based on the descriptions above and respond **only** with the exact product name.
    """
    
    response = model.generate_content(product_selection_prompt)

    selected_product = response.text.strip()
    if selected_product in SENTINELHUB_PRODUCTS:
        return selected_product
    return "Sentinel-2 L2A"

# Get Bounding Box
def get_bounding_box(location_name):
    location = geolocator.geocode(location_name)
    if location:
        lat, lon = location.latitude, location.longitude
        return BBox([lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1], CRS.WGS84)
    return None

# Get Available Dates
def get_available_dates(bbox, time_range, product):
    catalog = SentinelHubCatalog(config=config)
    selected_collection = product_mapping.get(product)
    results = catalog.search(collection=selected_collection, bbox=bbox, time=time_range)
    available_dates = sorted(set(item["properties"]["datetime"].split("T")[0] for item in results))

    return available_dates

# Generate Evalscript
def get_evalscript(product, analysis_type):
    available_bands = [band.name for band in (product_mapping.get(product).bands)]

    visualization_prompt = f"""
    Based on the given analysis type, generate a JavaScript visualization function for SentinelHub's evalscript.

    **Analysis Type:** "{analysis_type}"
    **Available Bands:** {", ".join(available_bands)}

    Provide a valid JavaScript function that makes use of the most relevant bands for this analysis.

    Ensure:
    - The function follows SentinelHub's `evaluatePixel()` structure.
    - No markdown formatting, no triple backticks, and no explanations.
    - Only return the function itself.

    **Example Output:**
    function evaluatePixel(sample) {{
        return [sample.B04, sample.B03, sample.B02]; // True color composite
    }}
    """

    response = model.generate_content(visualization_prompt)
    visualization_function = response.text.strip()

    # Clean any markdown formatting if it appears
    visualization_function = re.sub(r"^```(?:js|javascript)?\n|```$", "", visualization_function).strip()

    setup_prompt = f"""
    Based on the following JavaScript visualization function, generate a valid SentinelHub `setup()` function.

    **Visualization Function:**
    {visualization_function}

    Ensure:
    - The input includes only the bands used in `evaluatePixel()`.
    - The output band count matches the return format of `evaluatePixel()`.
    - The function follows SentinelHub's `setup()` structure.
    - No markdown formatting, no triple backticks, and no explanations.
    - Only return the function itself.

    **Example Output:**
    function setup() {{
        return {{
            input: [{{bands: ["B04", "B03", "B02"]}}],
            output: {{bands: 3}}
        }};
    }}
    """

    response_setup = model.generate_content(setup_prompt)
    setup_function = response_setup.text.strip()

    # Clean any markdown formatting
    setup_function = re.sub(r"^```(?:js|javascript)?\n|```$", "", setup_function).strip()

    evalscript = f"{setup_function}\n\n{visualization_function}"

    return evalscript


# Process Query
def process_query(user_query):
    parsed_query = extract_query_details(user_query)
    location = parsed_query.get("location", "Unknown")
    bbox = get_bounding_box(location)
    if not bbox:
        return "Could not determine bounding box."
    
    time_range = parsed_query.get("time_range", ["2024-01-01", "2025-01-01"])
    analysis_type = parsed_query.get("analysis_type", "NDVI trend")
    sentinel_product = select_sentinel_product(analysis_type)
    evalscript = get_evalscript(sentinel_product, analysis_type)
    available_dates = get_available_dates(bbox, time_range, sentinel_product)
    
    return {
        "location": location,
        "bounding_box": bbox,
        "time_range": time_range,
        "analysis_type": analysis_type,
        "sentinel_product": sentinel_product,
        "evalscript": evalscript,
        "available_dates": available_dates
    }

def explain_query_result(query,evalscript, result):
    explanation_prompt = f"""
    You have successfully retrieved SentinelHub data.
    Query: "{query}"

    Evalscript: "{evalscript}"

    Result: "{result}"
    Explain the retrieved images in simple terms.
    """
    explanation = model.generate_content(explanation_prompt).text.strip()
    return explanation

# Get Sentinel Request
def get_sentinel_request(time_interval, bbox, product, config, evalscript):
    bbox_width = bbox.upper_right[0] - bbox.lower_left[0]
    bbox_height = bbox.upper_right[1] - bbox.lower_left[1]
    aspect_ratio = bbox_width / bbox_height

    img_width = 2048
    img_height = int(img_width / aspect_ratio)

    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=product_mapping[product],
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=[img_width, img_height],
        config=config,
    )


st.title("🌍 Earth-LM")

# Example Queries
example_queries = [
    "Show flood detection in Jakarta last month",
    "Show agricultural data in Muzaffarnagar over the past year",
    "Show urban expansion in Dubai over the 6 months",
    "Show deforestation in the Amazon Rainforest in 2020"
]

# Dropdown for Example Queries
selected_query = st.selectbox("🔍 Choose an example query or enter your own:", ["Enter custom query"] + example_queries)

# Text Input for Custom Query
if selected_query == "Enter custom query":
    user_query = st.text_input("🔍 Enter your query:", "")
else:
    user_query = selected_query

if st.button("Process Query"):
    with st.spinner("🔄 Processing query..."):
        
        # Extract Query Details
        st.write("🔍 **Extracting query details...**")
        parsed_query = extract_query_details(user_query)

        if not parsed_query:
            st.error("❌ Could not parse query. Try refining your input.")
            st.stop()

        location = parsed_query.get("location", "Unknown")
        time_range = parsed_query.get("time_range", ["2024-01-01", "2025-01-01"])
        analysis_type = parsed_query.get("analysis_type", "NDVI trend")

        st.success(f"✅ **Extracted Query Details:**\n- 📍 Location: {location}\n- 📅 Time Range: {time_range}\n- 📊 Analysis Type: {analysis_type}")

        # Get Bounding Box
        st.write("🌍 **Finding location coordinates...**")
        bbox = get_bounding_box(location)
        
        if not bbox:
            st.error("❌ Could not determine the bounding box. Try entering a specific city or country.")
            st.stop()
        
        st.success("✅ Bounding box retrieved!")

        # Select Sentinel Product
        st.write("🛰 **Selecting best Sentinel product...**")
        sentinel_product = select_sentinel_product(analysis_type)
        st.success(f"✅ Selected Sentinel Product: **{sentinel_product}**")

        # Generate Evalscript
        st.write("🖥 **Generating Evalscript for visualization...**")
        evalscript = get_evalscript(sentinel_product, analysis_type)
        st.success("✅ Evalscript generated successfully!")

        # Get Available Dates
        st.write("📅 **Finding available satellite images...**")
        available_dates = get_available_dates(bbox, time_range, sentinel_product)

        if not available_dates:
            st.error("❌ No available satellite images for this time range. Try adjusting the time period.")
            st.stop()

        st.success(f"✅ Found {len(available_dates)} available image dates.")

        # Display Available Dates
        st.write("📆 **Available Image Dates:**")
        st.write(", ".join(available_dates))

        # Retrieve Sentinel Images for All Available Dates
        st.write(f"📡 **Fetching Sentinel data for {len(available_dates)} dates...**")

        images = []
        list_of_requests = [get_sentinel_request([date, date], bbox, sentinel_product, config, evalscript) for date in available_dates]
        list_of_requests = [request.download_list[0] for request in list_of_requests]
        data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5)
        

        images = list(zip(available_dates, data))

        # Show Retrieved Images
        st.write("🖼 **Retrieved Sentinel Images:**")
        for date, image in images:
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.set_title(f"Sentinel Image - {date}")
            ax.axis("off")
            st.pyplot(fig)

        # Explain Query Results
        explanation = explain_query_result(parsed_query, evalscript, images)
        st.write("📖 **Analysis Explanation:**")
        st.info(explanation)