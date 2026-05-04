
# ==========================================
# FINAL STREAMLIT APP - NO ERRORS
# ==========================================

# Install Streamlit if not already installed
try:
    import streamlit as st
except ImportError:
 import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------------------
# 1. App Config (MUST BE FIRST)
# ------------------------------------------
st.set_page_config(page_title="YouTube Revenue Predictor", layout="wide")

# ------------------------------------------
# 2. Add YouTube Background Logic
# ------------------------------------------
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png");
             background-attachment: fixed;
             background-size: 300px; /* Controls the width of the logo */
             background-repeat: no-repeat;
             background-position: center;
             background-color: rgba(120, 120, 120, 0.95); /* Optional: slight overlay for readability */
         }}

         /* This ensures your content blocks stay readable over the background */
         .stMarkdown, .stDataFrame, .stTable, .stButton {{
             background-color: rgba(120, 120, 120, 0.8);
             padding: 10px;
             border-radius: 10px;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# ------------------------------------------
# 3. Load Files
# ------------------------------------------
# Wrap in try-except to prevent crash if files are missing
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please ensure .pkl files are in the directory.")
    st.stop()

# ------------------------------------------
# 4. App Content
# ------------------------------------------
st.title("📊 YouTube Ad Revenue Prediction")

# ------------------------------------------
# Sidebar Inputs
# ------------------------------------------
st.sidebar.header("📥 Enter Video Details")

views = st.sidebar.number_input("Views", value=10000)
likes = st.sidebar.number_input("Likes", value=1000)
comments = st.sidebar.number_input("Comments", value=200)
watch_time = st.sidebar.number_input("Watch Time (minutes)", value=30000)
video_length = st.sidebar.number_input("Video Length (minutes)", value=10)
subscribers = st.sidebar.number_input("Subscribers", value=500000)

# Dropdowns (must match training categories)
category = st.sidebar.selectbox("Category", ["Music", "Gaming", "Tech", "Entertainment", "Lifestyle"])
device = st.sidebar.selectbox("Device", ["Mobile", "Tablet", "TV"])
country = st.sidebar.selectbox("Country", ["US", "IN", "UK", "CA", "DE"])

# ------------------------------------------
# Feature Engineering
# ------------------------------------------
engagement_rate = (likes + comments) / views if views != 0 else 0
like_rate = likes / views if views != 0 else 0
comment_rate = comments / views if views != 0 else 0
watch_time_per_view = watch_time / views if views != 0 else 0
watch_efficiency = watch_time / video_length if video_length != 0 else 0
subs_engagement = views / subscribers if subscribers != 0 else 0

# ------------------------------------------
# Create Input DataFrame (MATCH TRAINING)
# ------------------------------------------
input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0

# Fill numerical features
input_df['views'] = views
input_df['likes'] = likes
input_df['comments'] = comments
input_df['watch_time_minutes'] = watch_time
input_df['video_length_minutes'] = video_length
input_df['subscribers'] = subscribers

# Engineered features
input_df['engagement_rate'] = engagement_rate
input_df['like_rate'] = like_rate
input_df['comment_rate'] = comment_rate
input_df['watch_time_per_view'] = watch_time_per_view
input_df['watch_efficiency'] = watch_efficiency
input_df['subs_engagement'] = subs_engagement

# ------------------------------------------
# Apply Encoding (Dynamic)
# ------------------------------------------
cat_col = f"category_{category}"
dev_col = f"device_{device}"
country_col = f"country_{country}"

if cat_col in input_df.columns:
    input_df[cat_col] = 1

if dev_col in input_df.columns:
    input_df[dev_col] = 1

if country_col in input_df.columns:
    input_df[country_col] = 1

# ------------------------------------------
# Scaling
# ------------------------------------------
input_scaled = scaler.transform(input_df)

# ------------------------------------------
# Prediction
# ------------------------------------------
if st.sidebar.button("Predict Revenue"):

    prediction = model.predict(input_scaled)

    st.subheader("💰 Predicted Revenue")
    st.success(f"${prediction[0]:.2f}")

# ------------------------------------------
# Debug Section (Optional)
# ------------------------------------------
with st.expander("🔍 Debug Info"):
    st.write("Input DataFrame:", input_df)
    st.write("Scaled Input:", input_scaled)
