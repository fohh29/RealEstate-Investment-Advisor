# script
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set Plot directory
PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def find_data():
    paths = ["india_housing_prices.csv", "data/raw/india_housing_prices.csv", "../india_housing_prices.csv"]
    for path in paths:
        if os.path.exists(path):
            return path
    return None

data_path = find_data()
if not data_path:
    print("❌ Error: 'india_housing_prices.csv' not found. Please ensure it is in the main project folder.")
    exit()

df = pd.read_csv(data_path)

def save_plot(filename):
    plt.savefig(f"{PLOT_DIR}/{filename}")
    plt.clf()
    plt.close()

def generate_eda():
    print(f"📊 Loading data from: {data_path}")
    print("🚀 Force-Generating all 20 Insights...")
    sns.set_style("whitegrid")
    
    # 1-5: Price & Size
    plt.figure(); sns.histplot(df['Price_in_Lakhs'], bins=30, kde=True); plt.title("Q1: Price Distribution"); save_plot("q1_price_dist.png")
    plt.figure(); sns.histplot(df['Size_in_SqFt'], bins=30, color='green'); plt.title("Q2: Size Distribution"); save_plot("q2_size_dist.png")
    plt.figure(); sns.barplot(x='Property_Type', y='Price_per_SqFt', data=df); plt.title("Q3: Price/SqFt by Type"); save_plot("q3_price_type.png")
    plt.figure(); sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', data=df, alpha=0.3); plt.title("Q4: Size vs Price"); save_plot("q4_size_price_scatter.png")
    plt.figure(); sns.boxplot(x=df['Price_in_Lakhs']); plt.title("Q5: Price Outliers"); save_plot("q5_price_outliers.png")

    # 6-10: Location
    plt.figure(figsize=(12,6)); df.groupby('State')['Price_per_SqFt'].mean().sort_values().plot(kind='barh'); plt.title("Q6: Price by State"); save_plot("q6_state_price.png")
    plt.figure(); df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10).plot(kind='bar'); plt.title("Q7: Price by City"); save_plot("q7_city_price.png")
    plt.figure(); df.groupby('Locality')['Age_of_Property'].median().head(5).plot(kind='pie', autopct='%1.1f%%'); plt.title("Q8: Age by Locality"); save_plot("q8_age_locality.png")
    plt.figure(); sns.countplot(x='BHK', data=df); plt.title("Q9: BHK Distribution"); save_plot("q9_bhk_dist.png")
    plt.figure(); top_loc = df['Locality'].value_counts().head(5).index; sns.boxplot(x='Locality', y='Price_in_Lakhs', data=df[df['Locality'].isin(top_loc)]); plt.title("Q10: Top Localities"); save_plot("q10_loc_trends.png")

    # 11-15: Relationships & Heatmap
    plt.figure(figsize=(12,10)); sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm'); plt.title("Q11: Correlation Heatmap"); save_plot("q11_heatmap.png")
    plt.figure(); sns.lineplot(x='Nearby_Schools', y='Price_per_SqFt', data=df); plt.title("Q12: Schools vs Price"); save_plot("q12_schools_impact.png")
    plt.figure(); sns.lineplot(x='Nearby_Hospitals', y='Price_in_Lakhs', data=df); plt.title("Q13: Hospitals vs Price"); save_plot("q13_hospitals_impact.png")
    plt.figure(); sns.violinplot(x='Furnished_Status', y='Price_in_Lakhs', data=df); plt.title("Q14: Furnishing vs Price"); save_plot("q14_furnish_price.png")
    plt.figure(); sns.barplot(x='Facing', y='Price_per_SqFt', data=df); plt.title("Q15: Price by Facing"); save_plot("q15_facing_price.png")

    # 16-20: Amenities & Ownership
    plt.figure(); df['Owner_Type'].value_counts().plot(kind='bar'); plt.title("Q16: Owner Types"); save_plot("q16_owner_dist.png")
    plt.figure(); df['Availability_Status'].value_counts().plot(kind='pie', autopct='%1.1f%%'); plt.title("Q17: Availability"); save_plot("q17_availability.png")
    plt.figure(); sns.boxplot(x='Parking_Space', y='Price_in_Lakhs', data=df); plt.title("Q18: Parking Impact"); save_plot("q18_parking_price.png")
    plt.figure(); sns.barplot(x='Amenities', y='Price_per_SqFt', data=df); plt.title("Q19: Amenities Impact"); save_plot("q19_amenities_impact.png")
    plt.figure()
    # Using a bar plot instead of regplot to avoid the DType error
    sns.barplot(x='Public_Transport_Accessibility', y='Price_per_SqFt', data=df)
    plt.title("Q20: Transport Accessibility vs Price")
    save_plot("q20_transport_impact.png")
    print(f"✅ DONE! All 20 files are in {PLOT_DIR}")

if __name__ == "__main__":
    generate_eda()