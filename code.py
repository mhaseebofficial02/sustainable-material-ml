import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Sustainable Materials & Alloys Recommender", layout="wide")

# Load datasets
materials_df = pd.read_excel("Sustainable_Materials_Database_RealNames_200Plus (1).xlsx")
alloys_df = pd.read_excel("Engineering_Alloys_Application_Database (1).xlsx")

# Region adjustment factors
region_factors = {
    "Global Average": 1.00,
    "Europe": 0.85,
    "North America": 1.10,
    "Asia": 1.30,
    "Africa": 1.20,
    "South America": 0.95,
    "Oceania": 1.05,
}

st.sidebar.title("📊 Choose Recommender")
mode = st.sidebar.radio("Select View:", ["Sustainable Materials", "Engineering Alloys"])
region = st.sidebar.selectbox("🌍 Select Region for CO₂ Adjustment", list(region_factors.keys()))
co2_factor = region_factors[region]

# --------------------- MATERIAL SECTION ---------------------
if mode == "Sustainable Materials":
    st.title("🌿 Sustainable Materials Recommender")

    filter_printable = st.checkbox("✅ Show only 3D printable materials")
    filtered_df = materials_df if not filter_printable else materials_df[materials_df["3D Printable"] == "Yes"]
    selected_material = st.selectbox("🔍 Choose a material:", filtered_df["Material"])

    if st.button("Suggest Greener Alternatives"):
        df = materials_df.copy()

        # Calculate Toxicity and Biodegradability Scores
        df["Toxicity Score"] = df["Toxicity"].map({"Low": 1, "Medium": 0.5, "High": 0})
        df["Biodegradability Score"] = df["Biodegradability"].map({"Yes": 1, "Partial": 0.5, "No": 0})

        # Using available CO₂ Footprint for LCA CO₂
        df["Adjusted CO₂"] = df["CO₂ Footprint (kg CO₂/kg)"] * co2_factor

        # Create sustainability score with LCA data
        df["Sustainability Score with LCA (0-100)"] = 100 - (df["Adjusted CO₂"] / df["Adjusted CO₂"].max()) * 100

        features = df[["Adjusted CO₂", "Energy Intensity (MJ/kg)", "Recyclability (%)", "Toxicity Score", "Biodegradability Score"]]
        scaled = MinMaxScaler().fit_transform(features)

        idx = df[df["Material"] == selected_material].index[0]
        base_score = df["Sustainability Score with LCA (0-100)"][idx]
        sims = cosine_similarity([scaled[idx]], scaled)[0]
        candidates = [i for i in range(len(df)) if df["Sustainability Score with LCA (0-100)"][i] > base_score and i != idx]
        top = sorted(candidates, key=lambda i: -sims[i])[:3]

        st.success("🌱 Greener Alternatives:")
        st.dataframe(df.iloc[top][["Material", "Category", "Sustainability Score with LCA (0-100)", "3D Printable", "Printing Notes"]])

        row = df.iloc[idx]
        st.header(f"📊 Sustainability Profile: {selected_material}")
        st.write(f"**LCA CO₂ Total (Adjusted for {region}):** {round(row['Adjusted CO₂'], 2)} kg CO₂/kg")
        st.write(f"**Energy Intensity:** {row['Energy Intensity (MJ/kg)']} MJ/kg")
        st.write(f"**Recyclability:** {row['Recyclability (%)']}%")
        st.write(f"**Toxicity:** {row['Toxicity']}")
        st.write(f"**Biodegradability:** {row['Biodegradability']}")
        st.write(f"**3D Printable:** {row['3D Printable']}")
        st.write(f"**Printing Notes:** {row['Printing Notes']}")

        st.subheader("♻️ Sustainability Score Meaning:")
        if base_score <= 40:
            st.error("❌ Poor sustainability")
        elif base_score <= 60:
            st.warning("⚠️ Moderate sustainability")
        elif base_score <= 80:
            st.success("✅ Good sustainability")
        else:
            st.balloons()
            st.success("🌟 Excellent sustainability")

        st.subheader("📈 CO₂ Savings Chart")
        try:
            best_alt = df.iloc[top[0]]
            current_co2 = row['Adjusted CO₂']
            alt_co2 = best_alt['Adjusted CO₂']
            percent = round((current_co2 - alt_co2) / current_co2 * 100, 2)
            fig, ax = plt.subplots()
            ax.bar(["Current", "Recommended"], [current_co2, alt_co2], color=["red", "green"])
            ax.set_ylabel("CO₂ Footprint (kg CO₂/kg)")
            ax.set_title(f"CO₂ Reduction Potential: {percent}%")
            st.pyplot(fig)

            # 10-Year CO₂ Savings Forecast
            annual_savings = current_co2 - alt_co2
            forecast_savings = annual_savings * 10
            st.subheader(f"🌍 10-Year CO₂ Savings Forecast")
            st.write(f"If you switch to {best_alt['Material']} from {row['Material']}, you could save approximately **{round(forecast_savings, 2)} kg CO₂** over the next 10 years.")

            # Visualization of the 10-Year CO₂ savings
            fig, ax = plt.subplots()
            ax.bar(["Current", "Recommended"], [current_co2 * 10, alt_co2 * 10], color=["red", "green"])
            ax.set_ylabel("CO₂ Emissions (kg CO₂ over 10 years)")
            ax.set_title(f"10-Year CO₂ Savings Forecast: {round(forecast_savings, 2)} kg CO₂")
            st.pyplot(fig)
        except:
            st.warning("⚠️ Could not generate chart.")

# --------------------- ALLOY SECTION ---------------------
elif mode == "Engineering Alloys":
    st.title("🛠️ Engineering Alloy Recommender")
    selected_alloy = st.selectbox("🔩 Choose an alloy:", alloys_df["Alloy Name"])
    row = alloys_df[alloys_df["Alloy Name"] == selected_alloy].iloc[0]

    st.subheader("📘 Alloy Profile:")
    st.write(f"**Family:** {row['Family']}")
    st.write(f"**Yield Strength:** {row['Yield Strength (MPa)']} MPa")
    st.write(f"**Density:** {row['Density (g/cm³)']} g/cm³")
    st.write(f"**Corrosion Resistance:** {row['Corrosion Resistance']}")
    st.write(f"**Cost Level:** {row['Cost']}")
    st.write(f"**Eco-Score:** {row['Eco-Score (0-100)']}")
    st.write(f"**Applications:** {row['Common Applications']}")
    st.write(f"**3D Printable:** {row['3D Printable']}")
    st.write(f"**Printing Notes:** {row['Printing Notes']}")

    eco_score = row['Eco-Score (0-100)']
    st.subheader("♻️ Sustainability Interpretation:")
    if eco_score <= 40:
        st.error("❌ Poor sustainability")
    elif eco_score <= 60:
        st.warning("⚠️ Moderate sustainability")
    elif eco_score <= 80:
        st.success("✅ Good sustainability")
    else:
        st.success("🌟 Excellent sustainability")

    st.subheader("📊 Eco-Score Comparison")
    better = alloys_df[alloys_df['Eco-Score (0-100)'] > eco_score]
    if not better.empty:
        alt = better.iloc[0]
        fig, ax = plt.subplots()
        ax.bar(["Selected", "Alternative"], [eco_score, alt['Eco-Score (0-100)']], color=["orange", "green"])
        ax.set_ylabel("Eco-Score (0-100)")
        ax.set_title(f"{selected_alloy} vs {alt['Alloy Name']}")
        st.pyplot(fig)
    else:
        st.info("No better eco-score alternative found.")
