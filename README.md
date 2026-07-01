# Human Development Index (HDI) Predictor

A web-based Machine Learning application that predicts a country's Human Development Index (HDI) score and classifies it into one of the four official development tiers (Very High, High, Medium, Low) based on national socioeconomic indicators.

---

## 1. Project Overview

The **Human Development Index (HDI) Predictor** utilizes a supervised machine learning pipeline to model the relationship between a nation's core development statistics and its official HDI rating. The application features:
- **ML Pipeline**: A Linear Regression model trained on synthetic, internally consistent country indicator datasets.
- **Web Interface**: A glassmorphic, responsive Flask web application that takes user inputs and runs the ML model to output prediction results.
- **Weakest Link Analysis**: Deterministic socioeconomic evaluation indicating which pillar (Health, Education, or standard of living/Income) acts as the primary constraint on the country's development index.

---

## 2. Technology Stack

- **Core Language**: Python 3.x
- **Data & Modeling**: NumPy, Pandas, Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Backend Service**: Flask
- **Frontend Layer**: HTML5, Vanilla CSS3 (Glassmorphism design, Outfit font typography), Jinja2 Templates
- **Model Serialization**: Pickle

---

## 3. Folder Structure

```
hdi-predictor/
├── dataset/
│   └── hdi_dataset.csv
├── model/
│   ├── train_model.py
│   ├── hdi_model.pkl
│   ├── encoders.pkl
│   └── metrics.json
├── notebooks/
│   ├── eda_and_training.ipynb
│   ├── create_notebook.py
│   └── run_eda.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/
│       ├── correlation_heatmap.png
│       ├── distributions.png
│       ├── life_expectancy_vs_hdi.png
│       └── schooling_vs_hdi.png
├── templates/
│   ├── home.html
│   ├── indexnew.html
│   ├── result.html
│   └── resultnew.html
├── app.py
├── HDI.pkl
├── requirements.txt
└── README.md
```

---

## 4. Dataset Description

The application dataset (`dataset/hdi_dataset.csv`) contains socioeconomic records for 195 countries. The features represent the core dimensions of the Human Development Index:
1. **Country**: Name of the nation (label-encoded during modeling).
2. **Life expectancy**: Average lifespan at birth in years (Range: 50.0 - 89.0).
3. **Mean years of schooling**: Average number of years of education received by people aged 25 and older (Range: 1.0 - 15.0).
4. **Gross national income (GNI) per capita**: Gross national income converted to international dollars using purchasing power parity (PPP) rates (Range: $293 - $120,000).
5. **Internet users**: Percentage of population using internet (correlated with economic strength, Range: 0.0% - 100.0%).
6. **Expected years of schooling**: Number of years of schooling a child of school entrance age can expect to receive.
7. **HDI Score**: Mathematically consistent output score (geometric mean of dimension indices).
8. **HDI Tier**: Classification labels based on official UNDP cutoffs (Low: < 0.550, Medium: 0.550 - 0.699, High: 0.700 - 0.799, Very High: >= 0.800).

---

## 5. Model Performance Metrics

A Linear Regression model was trained using an 80/20 train-test split on the 195-country indicator dataset. The exact evaluation metrics obtained on the test set are:

- **Coefficient of Determination ($R^2$ Score)**: `0.9674` (explains 96.74% of the variance)
- **Mean Absolute Error (MAE)**: `0.0299`
- **Root Mean Squared Error (RMSE)**: `0.0362`

These metrics confirm that the model learns a highly accurate representation of the underlying non-linear geometric relations via linear combinations of the indicators.

---

## 6. Setup and Execution Instructions

### Option A: Standard Pip Setup
1. Clone or copy the project workspace.
2. Open a terminal in the `/hdi-predictor/` directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the model training script (if you want to retrain the model and regenerate plots):
   ```bash
   python model/train_model.py
   python notebooks/run_eda.py
   ```
5. Run the web application:
   ```bash
   python app.py
   ```
6. Access the application in your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Option B: Anaconda Setup
1. Open Anaconda Prompt.
2. Create a new environment:
   ```bash
   conda create -n hdi-predictor python=3.10
   conda activate hdi-predictor
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   python app.py
   ```

---

## 7. Functional Scenarios and Verification

The application successfully models and validates the three required development tiers:

### Scenario 1: Very High Development
- **Input profile**: Australia (code 8) | Life Expectancy = 83.5 | Mean Schooling = 12.7 | GNI per Capita = $49,000 | Internet Users = 91%
- **Expected Outcome**: Predicted score $\ge 0.800$, classified as **Very High** tier.

### Scenario 2: Medium Development
- **Input profile**: India (code 76) | Life Expectancy = 67.2 | Mean Schooling = 6.7 | GNI per Capita = $6,800 | Internet Users = 43%
- **Expected Outcome**: Predicted score within $[0.550, 0.699]$, classified as **Medium** tier. The result page highlights *Income (Gross National Income)* or *Education* as the weakest dimension constraining the country.

### Scenario 3: Low Development
- **Input profile**: Afghanistan (code 0) | Life Expectancy = 53.8 | Mean Schooling = 3.0 | GNI per Capita = $1,800 | Internet Users = 18%
- **Expected Outcome**: Predicted score $< 0.550$, classified as **Low** tier, flagging severe developmental challenges and resource constraints.
