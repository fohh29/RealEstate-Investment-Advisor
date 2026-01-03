# Streamlit Application
import streamlit as st
import pandas as pd
import joblib
import os
import glob

# 1. Page Configuration
st.set_page_config(page_title="India Real Estate Advisor", layout="wide")

# 2. Load Models and Encoders
@st.cache_resource
def load_assets():
    folder = "models"
    assets = {
        "regressor": joblib.load(os.path.join(folder, "price_regressor.pkl")),
        "classifier": joblib.load(os.path.join(folder, "investment_classifier.pkl")),
        "city": joblib.load(os.path.join(folder, "city_encoder.pkl")),
        "state": joblib.load(os.path.join(folder, "state_encoder.pkl")),
        "type": joblib.load(os.path.join(folder, "property_type_encoder.pkl")),
        "furnish": joblib.load(os.path.join(folder, "furnished_status_encoder.pkl"))
    }
    return assets

try:
    a = load_assets()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("📍 Property Details")
city_input = st.sidebar.selectbox("City", a['city'].classes_)
state_input = st.sidebar.selectbox("State", a['state'].classes_)
prop_input = st.sidebar.selectbox("Property Type", a['type'].classes_)
furnish_input = st.sidebar.selectbox("Furnishing", a['furnish'].classes_)
sqft = st.sidebar.number_input("Size (Sq. Ft.)", value=1200)
bhk = st.sidebar.slider("BHK", 1, 5, 2)
price_now = st.sidebar.number_input("Current Price (Lakhs)", value=75.0)

# 4. MAIN UI START
st.title("🏢 India Real Estate Investment Predictor")

# --- tabs_
tab1, tab2 = st.tabs(["🚀 Prediction Tool", "📊 Market Insights"])

with tab1:
    st.subheader("Investment Potential Calculator")
    if st.button("Calculate Value", type="primary"):
        try:
            #  data dictionary matching model's expected features
            data = {
                'City': a['city'].transform([city_input])[0],
                'State': a['state'].transform([state_input])[0],
                'Property_Type': a['type'].transform([prop_input])[0],
                'Furnished_Status': a['furnish'].transform([furnish_input])[0],
                'Size_in_SqFt': sqft, 'BHK': bhk, 'Price_in_Lakhs': price_now,
                'Price_per_SqFt': (price_now * 100000) / sqft if sqft > 0 else 0,
                'Year_Built': 2020, 'Floor_No': 5, 'Total_Floors': 10, 'Age_of_Property': 5,
                'Nearby_Schools': 1, 'Nearby_Hospitals': 1, 'Public_Transport_Accessibility': 1,
                'Parking_Space': 1, 'Owner_Type': 0, 'Locality': 0, 'Security': 1, 
                'Amenities': 1, 'Facing': 0, 'Availability_Status': 0
            }
            
            df_input = pd.DataFrame([data])
            expected_cols = a['regressor'].feature_names_in_
            df_input = df_input[expected_cols]

            future_p = a['regressor'].predict(df_input)[0]
            is_good = a['classifier'].predict(df_input)[0]

            c1, c2 = st.columns(2)
            c1.metric("Predicted Price (5 Years)", f"₹{future_p:.2f} L")
            if is_good == 1:
                c2.success("✅ HIGH POTENTIAL")
                st.balloons()
            else:
                c2.warning("⚠️ LOW GROWTH POTENTIAL")
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.subheader("Exploratory Data Analysis (20 Insights)")
    
    #  display 20 plots:
    plot_files = [f"q{i}" for i in range(1, 21)]
    cols = st.columns(2)
    
    for index, plot_id in enumerate(plot_files):
        found_files = glob.glob(f"outputs/plots/{plot_id}_*.png")
        if found_files:
            target_col = cols[index % 2]
            target_col.image(found_files[0], use_container_width=True)
        elif os.path.exists(f"outputs/plots/{plot_id}.png"):
            target_col = cols[index % 2]
            target_col.image(f"outputs/plots/{plot_id}.png", use_container_width=True)