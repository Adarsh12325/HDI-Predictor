import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

# Initialize the Flask application
app = Flask(__name__)

# Load the model from the current directory (HDI.pkl) as shown in the user's screenshots
# If it fails, fallback to model/hdi_model.pkl
model_path = 'HDI.pkl'
if not os.path.exists(model_path):
    model_path = os.path.join('model', 'hdi_model.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model successfully loaded from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Validation Ranges
LIMITS = {
    'Life expectancy': (50.0, 89.0),
    'Mean years of schooling': (1.0, 15.0),
    'Gross national income (GNI) per capita': (293.0, 120000.0),
    'Internet users': (0.0, 100.0)
}

# Country labels for validation and UI mapping (from the LabelEncoder used in training)
COUNTRY_MAP = {
    "0": "Afghanistan",
    "8": "Australia",
    "13": "Bangladesh",
    "31": "Canada",
    "76": "India",
    "138": "Poland",
    "179": "Turkey"
}

def analyze_weakest_dimension(life_exp, mean_school, gni):
    """Computes UNDP index values for each dimension and identifies the weakest contributor."""
    # Health Index (LEI): (LE - 20) / (85 - 20)
    health_idx = (life_exp - 20.0) / (85.0 - 20.0)
    health_idx = np.clip(health_idx, 0.0, 1.0)
    
    # Education Index (EI): Mean years of schooling / 15 as proxy
    edu_idx = mean_school / 15.0
    edu_idx = np.clip(edu_idx, 0.0, 1.0)
    
    # Income Index (II): (ln(GNI) - ln(100)) / (ln(75000) - ln(100))
    gni_val = max(gni, 101.0) # avoid log of negative or <= 100
    income_idx = (np.log(gni_val) - np.log(100.0)) / (np.log(75000.0) - np.log(100.0))
    income_idx = np.clip(income_idx, 0.0, 1.0)
    
    indices = {
        "Health (Life Expectancy)": health_idx,
        "Education (Schooling)": edu_idx,
        "Income (Gross National Income)": income_idx
    }
    
    weakest_dim = min(indices, key=indices.get)
    weakest_score = indices[weakest_dim]
    
    explanations = {
        "Health (Life Expectancy)": (
            "Health & Longevity is the weakest dimension. Lower life expectancy (due to healthcare gaps, sanitation issues, "
            "or structural challenges) restricts overall human development outcomes."
        ),
        "Education (Schooling)": (
            "Education & Knowledge access is the weakest dimension. Low mean years of schooling indicates "
            "barriers to secondary or higher education, limiting skill development and workforce capability."
        ),
        "Income (Gross National Income)": (
            "Decent Standard of Living is the weakest dimension. Lower GNI per capita limits economic power, "
            "public infrastructure investment, and personal purchasing capability, dampening the development score."
        )
    }
    
    return weakest_dim, round(weakest_score, 3), explanations[weakest_dim]

@app.route('/')
def home():
    """Renders the landing home page."""
    return render_template('home.html')

@app.route('/Home', methods=['GET', 'POST'])
def my_home():
    """Alternative home route mirroring image routing."""
    return render_template('home.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    """Renders the predictor form page."""
    return render_template('indexnew.html', country_map=COUNTRY_MAP)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction request, processes input, runs Linear Regression, and returns results."""
    # Retrieve and validate inputs
    try:
        country_code = request.form.get('Country', '')
        life_exp_str = request.form.get('Life expectancy', '')
        mean_school_str = request.form.get('Mean years of schooling', '')
        gni_str = request.form.get('Gross national income (GNI) per capita', '')
        internet_str = request.form.get('Internet users', '')
        
        # Check for empty values
        if not country_code or not life_exp_str or not mean_school_str or not gni_str or not internet_str:
            return render_template('indexnew.html', 
                                   error="All fields are required. Please fill in all the forms.", 
                                   country_map=COUNTRY_MAP)
        
        country = float(country_code)
        life_exp = float(life_exp_str)
        mean_school = float(mean_school_str)
        gni = float(gni_str)
        internet = float(internet_str)
        
    except ValueError:
        return render_template('indexnew.html', 
                               error="Invalid numeric input. Please enter valid numbers for indicators.", 
                               country_map=COUNTRY_MAP)
    
    # Server-side validation ranges check
    validation_errors = []
    if not (LIMITS['Life expectancy'][0] <= life_exp <= LIMITS['Life expectancy'][1]):
        validation_errors.append(f"Life Expectancy must be between {LIMITS['Life expectancy'][0]} and {LIMITS['Life expectancy'][1]} years.")
    if not (LIMITS['Mean years of schooling'][0] <= mean_school <= LIMITS['Mean years of schooling'][1]):
        validation_errors.append(f"Mean Years of Schooling must be between {LIMITS['Mean years of schooling'][0]} and {LIMITS['Mean years of schooling'][1]} years.")
    if not (LIMITS['Gross national income (GNI) per capita'][0] <= gni <= LIMITS['Gross national income (GNI) per capita'][1]):
        validation_errors.append(f"GNI per capita must be between ${LIMITS['Gross national income (GNI) per capita'][0]} and ${LIMITS['Gross national income (GNI) per capita'][1]}.")
    if not (LIMITS['Internet users'][0] <= internet <= LIMITS['Internet users'][1]):
        validation_errors.append(f"Internet Users percentage must be between {LIMITS['Internet users'][0]}% and {LIMITS['Internet users'][1]}%.")
        
    if validation_errors:
        error_msg = " | ".join(validation_errors)
        return render_template('indexnew.html', error=error_msg, country_map=COUNTRY_MAP)

    # Perform prediction
    features_value = [np.array([country, life_exp, mean_school, gni, internet])]
    features_name = ['Country_Encoded', 'Life expectancy', 'Mean years of schooling', 'Gross national income (GNI) per capita', 'Internet users']
    df_pred = pd.DataFrame(features_value, columns=features_name)
    
    if model is None:
        return render_template('indexnew.html', error="Model error: Model pickle file not loaded.", country_map=COUNTRY_MAP)
        
    output = model.predict(df_pred)
    # The output is expected to be a 1D array or 2D array depending on train_model.py
    if hasattr(output, '__len__') and len(output.shape) > 1:
        y_pred = float(output[0][0])
    else:
        y_pred = float(output[0])
        
    # Cap predicted score between standard UNDP limits (0.3 to 0.96)
    y_pred = np.clip(y_pred, 0.30, 0.96)
    y_pred_rounded = round(y_pred, 2)
    
    # Determine classification tier and weak dimension explanation
    tier = ""
    if y_pred_rounded >= 0.80:
        tier = "Very High"
    elif y_pred_rounded >= 0.70:
        tier = "High"
    elif y_pred_rounded >= 0.55:
        tier = "Medium"
    else:
        tier = "Low"
        
    weak_dim, weak_score, explanation = analyze_weakest_dimension(life_exp, mean_school, gni)
    country_name = COUNTRY_MAP.get(country_code, "Selected Country")
    
    # Text format as expected by user's screenshot
    # Medium HDI 0.48 or Low HDI 0.35 etc.
    prediction_text = f"{tier} HDI {y_pred_rounded:.2f}"
    
    return render_template('result.html',
                           prediction_text=prediction_text,
                           y_pred=y_pred_rounded,
                           tier=tier,
                           weak_dim=weak_dim,
                           weak_score=weak_score,
                           explanation=explanation,
                           country_name=country_name,
                           life_exp=life_exp,
                           mean_school=mean_school,
                           gni=gni,
                           internet=internet)

if __name__ == '__main__':
    import os
    # Read dynamic port from environment or fallback to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
