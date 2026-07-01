import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Define a seed for reproducibility
np.random.seed(42)

# List of 195 real countries
countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Congo-Brazzaville)", "Costa Rica",
    "Croatia", "Cuba", "Cyprus", "Czechia", "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador",
    "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France",
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau",
    "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal",
    "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan",
    "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
    "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal",
    "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Korea",
    "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania",
    "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen",
    "Zambia", "Zimbabwe", "Kosovo", "Palestine", "Taiwan"
]
countries = countries[:195]  # Keep exactly 195

def generate_hdi_data():
    """Generates synthetic but realistic and mathematically consistent HDI data for 195 countries."""
    data = []
    for idx, country in enumerate(countries):
        # Generate raw indicator features with realistic ranges
        # High GNI per capita countries tend to have higher life expectancy and schooling
        if idx % 5 == 0:  # Very High development profile
            life_exp = np.random.uniform(78.0, 85.0)
            mean_school = np.random.uniform(11.0, 14.0)
            exp_school = np.random.uniform(13.0, 17.5)
            gni = np.random.uniform(35000, 115000)
            internet_pct = np.random.uniform(80.0, 99.0)
        elif idx % 5 in (1, 2):  # High and Medium development profile
            life_exp = np.random.uniform(65.0, 77.0)
            mean_school = np.random.uniform(6.0, 10.9)
            exp_school = np.random.uniform(9.0, 12.9)
            gni = np.random.uniform(5000, 30000)
            internet_pct = np.random.uniform(45.0, 79.0)
        else:  # Low development profile
            life_exp = np.random.uniform(50.0, 64.9)
            mean_school = np.random.uniform(1.5, 5.9)
            exp_school = np.random.uniform(3.0, 8.9)
            gni = np.random.uniform(300, 4900)
            internet_pct = np.random.uniform(5.0, 44.0)

        # Standard UNDP HDI Dimensional Index Calculations:
        # 1. Health Index (Life Expectancy Index)
        lei = (life_exp - 20.0) / (85.0 - 20.0)
        lei = np.clip(lei, 0.0, 1.0)
        
        # 2. Education Index
        mysi = mean_school / 15.0
        eysi = exp_school / 18.0
        ei = (mysi + eysi) / 2.0
        ei = np.clip(ei, 0.0, 1.0)
        
        # 3. Income Index
        ii = (np.log(gni) - np.log(100.0)) / (np.log(75000.0) - np.log(100.0))
        ii = np.clip(ii, 0.0, 1.0)
        
        # HDI Score is the geometric mean of the three dimensions
        hdi_score = (lei * ei * ii) ** (1.0 / 3.0)
        # Clip HDI score between 0.30 and 0.96 to be realistic
        hdi_score = np.clip(hdi_score, 0.30, 0.96)
        
        # Determine tier
        if hdi_score >= 0.800:
            tier = "Very High"
        elif hdi_score >= 0.700:
            tier = "High"
        elif hdi_score >= 0.550:
            tier = "Medium"
        else:
            tier = "Low"

        # Round values for realism
        data.append({
            "Country": country,
            "Life expectancy": round(life_exp, 1),
            "Mean years of schooling": round(mean_school, 1),
            "Gross national income (GNI) per capita": round(gni, 0),
            "Internet users": round(internet_pct, 1),
            "Expected years of schooling": round(exp_school, 1),
            "HDI Score": round(hdi_score, 3),
            "HDI Tier": tier
        })
        
    return pd.DataFrame(data)

def main():
    print("Generating synthetic HDI dataset...")
    df = generate_hdi_data()
    
    # Create dataset directory if not exists
    os.makedirs("dataset", exist_ok=True)
    df.to_csv("dataset/hdi_dataset.csv", index=False)
    print("Dataset saved to dataset/hdi_dataset.csv")

    # Fit Country Label Encoder
    print("Label encoding categorical variables...")
    le_country = LabelEncoder()
    df["Country_Encoded"] = le_country.fit_transform(df["Country"])
    
    # Save the encoder for deployment usage
    os.makedirs("model", exist_ok=True)
    with open("model/encoders.pkl", "wb") as f:
        pickle.dump(le_country, f)
    print("Country LabelEncoder saved to model/encoders.pkl")

    # Select independent (X) and dependent (Y) variables
    # Independent variables match what's used in the training code images
    # We map 'Country' using the encoded values
    X = df[["Country_Encoded", "Life expectancy", "Mean years of schooling", "Gross national income (GNI) per capita", "Internet users"]].copy()
    Y = df["HDI Score"]

    # Implement check nulls & imputation logic
    print("Validating missing value imputation logic...")
    # Add a deliberately nulled test row for assertion testing
    test_null_row = X.iloc[0].copy()
    test_null_row["Life expectancy"] = np.nan
    X_test_null = pd.DataFrame([test_null_row])
    
    # Apply imputation: check nulls and fill with mean
    nulls_count = X_test_null.isnull().sum().sum()
    assert nulls_count > 0, "Test setup error: Null row must contain at least one null value."
    
    # Impute missing values using training feature means
    X_imputed = X_test_null.fillna(X.mean())
    assert not X_imputed.isnull().any().any(), "Imputation failed: Null values remain."
    print("Imputation logic assertion passed! Null values successfully filled with column means.")

    # Apply mean imputation to dataset X for robustness
    X = X.fillna(X.mean())

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {
        "R2_Score": round(r2, 4),
        "Mean_Absolute_Error": round(mae, 4),
        "Root_Mean_Squared_Error": round(rmse, 4)
    }

    print("Model Evaluation Metrics:")
    print(f"  R2 Score: {metrics['R2_Score']}")
    print(f"  Mean Absolute Error: {metrics['Mean_Absolute_Error']}")
    print(f"  Root Mean Squared_Error: {metrics['Root_Mean_Squared_Error']}")

    # Save metrics to json
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to model/metrics.json")

    # Serialize model to pickle
    with open("model/hdi_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to model/hdi_model.pkl")

    # Duplicate model to root directory as HDI.pkl to match app.py loading in the images
    with open("HDI.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model duplicated to root directory as HDI.pkl for app.py accessibility")

if __name__ == "__main__":
    main()
