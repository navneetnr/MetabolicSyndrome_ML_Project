import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Metabolic Syndrome Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        padding: 0.5rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-moderate {
        background-color: #ffd43b;
        padding: 1rem;
        border-radius: 10px;
        color: #2c3e50;
        text-align: center;
    }
    .risk-low {
        background-color: #51cf66;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Metabolic Syndrome Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["📈 Data Analysis", "🤖 Model Training & Evaluation", "🔮 Single Prediction", "📚 About"])

# Load data function with caching
@st.cache_data
def load_data():
    csv_path = r"C:\Users\LENOVO\Desktop\Sem 6\ML\Metabolic_Syndrome.csv"
    
    try:
        df = pd.read_csv(csv_path)
        st.sidebar.success(f"✅ Dataset loaded successfully!")
        st.sidebar.info(f"Total samples: {df.shape[0]} | Features: {df.shape[1]}")
        return df
    except FileNotFoundError:
        st.sidebar.error(f"❌ File not found at: {csv_path}")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {e}")
        return None

# Data preprocessing function
@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    
    # Data cleaning
    df_processed["Marital"] = df_processed["Marital"].fillna(df_processed["Marital"].mode()[0])
    df_processed["Income"] = df_processed["Income"].fillna(df_processed["Income"].median())
    df_processed["WaistCirc"] = df_processed["WaistCirc"].fillna(df_processed["WaistCirc"].median())
    df_processed["BMI"] = df_processed["BMI"].fillna(df_processed["BMI"].median())
    
    # One-hot encoding
    df_processed = pd.get_dummies(df_processed, columns=["Sex", "Race"], drop_first=True, dtype=int)
    
    # Label encoding for Marital
    le = LabelEncoder()
    df_processed["Marital"] = le.fit_transform(df_processed["Marital"])
    
    return df_processed, le

# Function to train models
@st.cache_resource
def train_models(x_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="saga", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=42, probability=True),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False)
    }
    
    trained_models = {}
    progress_bar = st.progress(0)
    for i, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name}..."):
            model.fit(x_train, y_train)
            trained_models[name] = model
            progress_bar.progress((i + 1) / len(models))
    progress_bar.empty()
    
    return trained_models

# Function to analyze Metabolic Syndrome risk factors
def analyze_risk_factors(age, waist_circ, bmi, hdl, triglycerides, sex):
    risk_factors = []
    risk_score = 0
    
    # Waist Circumference risk
    if sex == "Male":
        if waist_circ > 102:
            risk_factors.append("⚠️ **High Waist Circumference** (>102 cm - High Risk)")
            risk_score += 2
        elif waist_circ > 94:
            risk_factors.append("⚠️ **Increased Waist Circumference** (>94 cm - Moderate Risk)")
            risk_score += 1
    else:
        if waist_circ > 88:
            risk_factors.append("⚠️ **High Waist Circumference** (>88 cm - High Risk)")
            risk_score += 2
        elif waist_circ > 80:
            risk_factors.append("⚠️ **Increased Waist Circumference** (>80 cm - Moderate Risk)")
            risk_score += 1
    
    # Triglycerides risk
    if triglycerides > 200:
        risk_factors.append("⚠️ **Very High Triglycerides** (>200 mg/dL - High Risk)")
        risk_score += 2
    elif triglycerides > 150:
        risk_factors.append("⚠️ **High Triglycerides** (>150 mg/dL - Moderate Risk)")
        risk_score += 1
    
    # HDL cholesterol risk
    if sex == "Male":
        if hdl < 40:
            risk_factors.append("⚠️ **Low HDL Cholesterol** (<40 mg/dL - High Risk)")
            risk_score += 2
        elif hdl < 50:
            risk_factors.append("⚠️ **Borderline HDL** (<50 mg/dL - Moderate Risk)")
            risk_score += 1
    else:
        if hdl < 50:
            risk_factors.append("⚠️ **Low HDL Cholesterol** (<50 mg/dL - High Risk)")
            risk_score += 2
        elif hdl < 60:
            risk_factors.append("⚠️ **Borderline HDL** (<60 mg/dL - Moderate Risk)")
            risk_score += 1
    
    # BMI risk
    if bmi >= 35:
        risk_factors.append("⚠️ **Severe Obesity** (BMI ≥35 - High Risk)")
        risk_score += 2
    elif bmi >= 30:
        risk_factors.append("⚠️ **Obese** (BMI 30-34.9 - Moderate Risk)")
        risk_score += 1
    elif bmi >= 25:
        risk_factors.append("⚠️ **Overweight** (BMI 25-29.9 - Mild Risk)")
        risk_score += 0.5
    
    # Age risk
    if age >= 60:
        risk_factors.append("⚠️ **Advanced Age** (>60 years - Increased Risk)")
        risk_score += 1
    elif age >= 45:
        risk_factors.append("⚠️ **Middle Age** (45-60 years - Moderate Risk)")
        risk_score += 0.5
    
    return risk_factors, risk_score

# Load data
df = load_data()

# Stop if no data loaded
if df is None:
    st.stop()

# Process the data
df_processed, marital_encoder = preprocess_data(df)

# Prepare features and target
X = df_processed.drop("MetabolicSyndrome", axis=1)
y = df_processed["MetabolicSyndrome"]

# Get numerical columns for scaling
num_cols = ["Age", "Income", "WaistCirc", "BMI", "HDL", "Triglycerides"]

# ==================== DATA ANALYSIS PAGE ====================
if page == "📈 Data Analysis":
    st.markdown('<h2 class="sub-header">📊 Dataset Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        metabolic_count = df['MetabolicSyndrome'].sum()
        st.metric("Metabolic Syndrome Cases", metabolic_count)
    with col4:
        metabolic_pct = (metabolic_count / df.shape[0]) * 100
        st.metric("Prevalence", f"{metabolic_pct:.1f}%")
    
    st.markdown("---")
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10))
    
    # Dataset info
    st.subheader("ℹ️ Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values:**")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum()
        })
        st.dataframe(missing_df[missing_df['Missing Values'] > 0])
    
    with col2:
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values
        })
        st.dataframe(dtype_df)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe())
    
    # Target distribution
    st.subheader("🎯 Target Variable Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=df['MetabolicSyndrome'], ax=ax, palette=['#51cf66', '#ff6b6b'])
        ax.set_title('Metabolic Syndrome Distribution', fontsize=14)
        ax.set_xlabel('Metabolic Syndrome (0 = No, 1 = Yes)')
        ax.set_ylabel('Count')
        for i, v in enumerate(df['MetabolicSyndrome'].value_counts().values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#51cf66', '#ff6b6b']
        df['MetabolicSyndrome'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=colors, labels=['No Metabolic Syndrome', 'Metabolic Syndrome'])
        ax.set_title('Target Variable Distribution', fontsize=14)
        ax.set_ylabel('')
        st.pyplot(fig)
    
    # Numerical features distribution
    st.subheader("📈 Numerical Features Distribution")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribution of {col}', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    st.pyplot(fig)
    
    # Compare distributions by Metabolic Syndrome
    st.subheader("📊 Feature Comparison: With vs Without Metabolic Syndrome")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        df_no = df[df['MetabolicSyndrome'] == 0][col].dropna()
        df_yes = df[df['MetabolicSyndrome'] == 1][col].dropna()
        axes[i].hist(df_no, bins=15, alpha=0.5, label='No Metabolic Syndrome', color='#51cf66')
        axes[i].hist(df_yes, bins=15, alpha=0.5, label='Has Metabolic Syndrome', color='#ff6b6b')
        axes[i].set_title(f'{col} Distribution', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    st.pyplot(fig)

# ==================== MODEL TRAINING PAGE ====================
elif page == "🤖 Model Training & Evaluation":
    st.markdown('<h2 class="sub-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.subheader("⚙️ Training Parameters")
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
    apply_smote = st.sidebar.checkbox("Apply SMOTE (Balance Dataset)", value=True)
    train_button = st.sidebar.button("🚀 Train Models", type="primary")
    
    if train_button:
        with st.spinner("Training models... This may take a moment..."):
            X_data = df_processed.drop("MetabolicSyndrome", axis=1)
            y_data = df_processed["MetabolicSyndrome"]
            
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=42)
            
            if apply_smote:
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
            
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
            
            models = train_models(X_train, y_train)
            
            results = []
            
            st.success("✅ Models trained successfully!")
            
            tabs = st.tabs(["📊 Performance Comparison", "🔍 Individual Model Results", "📈 Training Visualization"])
            
            with tabs[0]:
                st.subheader("Model Performance Comparison")
                
                for name, model in models.items():
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    results.append({
                        "Model": name,
                        "Accuracy": f"{accuracy*100:.2f}%",
                        "Recall": f"{recall:.4f}",
                        "F1 Score": f"{f1:.4f}"
                    })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics = ['Accuracy', 'Recall', 'F1 Score']
                x = np.arange(len(models))
                width = 0.25
                
                for i, metric in enumerate(metrics):
                    values = []
                    for j in range(len(models)):
                        val = results_df[metric].iloc[j]
                        if metric == 'Accuracy':
                            values.append(float(val.replace('%', '')))
                        else:
                            values.append(float(val))
                    ax.bar(x + i*width, values, width, label=metric)
                
                ax.set_xlabel('Models')
                ax.set_ylabel('Score')
                ax.set_title('Model Performance Comparison')
                ax.set_xticks(x + width)
                ax.set_xticklabels(list(models.keys()), rotation=45, ha='right')
                ax.legend()
                ax.set_ylim(0, 1)
                plt.tight_layout()
                st.pyplot(fig)
            
            with tabs[1]:
                st.subheader("Individual Model Analysis")
                
                for name, model in models.items():
                    with st.expander(f"📊 {name}"):
                        y_pred = model.predict(X_test)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
                        with col2:
                            st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
                        with col3:
                            st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
                        
                        fig, ax = plt.subplots(figsize=(6, 5))
                        cm = confusion_matrix(y_test, y_pred)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title(f'{name} - Confusion Matrix')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                        
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df)
            
            with tabs[2]:
                st.subheader("Training Data Distribution")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(x=y_data, ax=ax, palette=['#51cf66', '#ff6b6b'])
                    ax.set_title('Original Dataset Distribution')
                    st.pyplot(fig)
                
                with col2:
                    if apply_smote:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.countplot(x=y_train, ax=ax, palette=['#51cf66', '#ff6b6b'])
                        ax.set_title('Balanced Dataset (After SMOTE)')
                        st.pyplot(fig)
                    else:
                        st.info("SMOTE was not applied. Dataset is imbalanced.")
                
                if "Random Forest" in models:
                    st.subheader("🌲 Random Forest Feature Importance")
                    rf_model = models["Random Forest"]
                    feature_importance = pd.DataFrame({
                        'Feature': X_data.columns,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
                    ax.set_title('Top 10 Feature Importance - Random Forest')
                    plt.tight_layout()
                    st.pyplot(fig)
    
    else:
        st.info("👈 Click 'Train Models' in the sidebar to start training!")

# ==================== SINGLE PREDICTION PAGE (FIXED) ====================
elif page == "🔮 Single Prediction":
    st.markdown('<h2 class="sub-header">🔮 Metabolic Syndrome Risk Assessment</h2>', unsafe_allow_html=True)
    
    st.write("Enter patient information to assess Metabolic Syndrome risk.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Personal Information")
        age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=45.0, step=1.0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        race = st.selectbox("Race", ["White", "Black", "Hispanic", "Asian", "Other"])
        income = st.number_input("Annual Income (USD)", min_value=0.0, max_value=500000.0, value=50000.0, step=5000.0)
    
    with col2:
        st.subheader("📏 Clinical Measurements")
        waist_circ = st.number_input("Waist Circumference (cm)", min_value=50.0, max_value=200.0, value=90.0, step=1.0)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=25.0, step=0.1, format="%.1f")
        hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20.0, max_value=150.0, value=50.0, step=1.0)
        triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50.0, max_value=500.0, value=150.0, step=5.0)
    
    st.markdown("---")
    
    # Quick Risk Analysis Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Metabolic Syndrome Risk", type="primary", use_container_width=True)
    
    if analyze_button:
        # Perform risk factor analysis
        risk_factors, risk_score = analyze_risk_factors(age, waist_circ, bmi, hdl, triglycerides, sex)
        
        # Display risk assessment
        st.markdown("---")
        st.subheader("📋 Metabolic Syndrome Risk Assessment Results")
        
        # Risk level determination
        if risk_score >= 4:
            risk_level = "HIGH RISK"
            risk_color = "risk-high"
            risk_icon = "🔴"
            recommendation = "Please consult a healthcare provider immediately for comprehensive evaluation."
        elif risk_score >= 2:
            risk_level = "MODERATE RISK"
            risk_color = "risk-moderate"
            risk_icon = "🟡"
            recommendation = "Consider lifestyle modifications and consult a healthcare provider for screening."
        else:
            risk_level = "LOW RISK"
            risk_color = "risk-low"
            risk_icon = "🟢"
            recommendation = "Maintain healthy lifestyle habits. Regular check-ups recommended."
        
        # Display risk level
        st.markdown(f"""
        <div class="{risk_color}" style="padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h3 style="margin: 0;">{risk_icon} {risk_level}</h3>
            <p style="margin: 0.5rem 0 0 0;">Risk Score: {risk_score}/8</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display risk factors
        if risk_factors:
            st.subheader("⚠️ Identified Risk Factors")
            for rf in risk_factors:
                st.write(rf)
        else:
            st.success("✅ No major risk factors identified!")
        
        # Train model and get prediction
        with st.spinner("Running ML prediction..."):
            # Prepare data for prediction - ENCODE CATEGORICAL VARIABLES
            # Create a dictionary with numerical values
            input_data = {
                "Age": float(age),
                "Income": float(income),
                "WaistCirc": float(waist_circ),
                "BMI": float(bmi),
                "HDL": float(hdl),
                "Triglycerides": float(triglycerides)
            }
            
            # Encode Marital status (same as training)
            marital_mapping = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
            input_data["Marital"] = marital_mapping[marital]
            
            # Add one-hot encoded columns for Sex
            input_data["Sex_Male"] = 1 if sex == "Male" else 0
            
            # Add one-hot encoded columns for Race
            input_data["Race_Hispanic"] = 1 if race == "Hispanic" else 0
            input_data["Race_Other"] = 1 if race == "Other" else 0
            input_data["Race_White"] = 1 if race == "White" else 0
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all columns from training are present
            X_data = df_processed.drop("MetabolicSyndrome", axis=1)
            for col in X_data.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[X_data.columns]
            
            # Scale numerical features
            scaler = StandardScaler()
            X_train_full, _, y_train_full, _ = train_test_split(X_data, y, test_size=0.2, random_state=42)
            X_train_full[num_cols] = scaler.fit_transform(X_train_full[num_cols])
            input_df[num_cols] = scaler.transform(input_df[num_cols])
            
            # Train and predict
            rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_final.fit(X_train_full, y_train_full)
            
            prediction = rf_final.predict(input_df)[0]
            probability = rf_final.predict_proba(input_df)[0]
        
        st.subheader("🤖 Machine Learning Prediction")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **ML Prediction: Metabolic Syndrome LIKELY**")
                st.info(f"Confidence: {probability[1]*100:.1f}%")
            else:
                st.success("✅ **ML Prediction: Metabolic Syndrome UNLIKELY**")
                st.info(f"Confidence: {probability[0]*100:.1f}%")
        
        with col2:
            st.metric("Probability of No Metabolic Syndrome", f"{probability[0]*100:.1f}%")
            st.metric("Probability of Metabolic Syndrome", f"{probability[1]*100:.1f}%")
        
        # Recommendations
        st.subheader("📝 Recommendations")
        st.write(recommendation)
        
        if risk_score >= 2:
            with st.expander("💡 Lifestyle Modification Tips"):
                st.markdown("""
                - **Increase Physical Activity**: Aim for 30 minutes of moderate exercise daily
                - **Healthy Diet**: Reduce saturated fats, sugars, and refined carbohydrates
                - **Weight Management**: Even 5-10% weight loss can significantly reduce risk
                - **Regular Monitoring**: Check blood pressure, blood sugar, and cholesterol regularly
                - **Stress Management**: Practice relaxation techniques like meditation or yoga
                """)

# ==================== ABOUT PAGE ====================
else:
    st.markdown('<h2 class="sub-header">📚 About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🏥 Metabolic Syndrome Prediction System
    
    This application uses machine learning to predict whether a patient has Metabolic Syndrome based on various health and demographic factors.
    
    ### 📊 What is Metabolic Syndrome?
    
    Metabolic syndrome is a cluster of conditions that occur together, increasing your risk of heart disease, stroke, and type 2 diabetes.
    
    ### 🤖 Models Used
    
    The system implements four different machine learning algorithms:
    
    1. **Logistic Regression** - A statistical model for binary classification
    2. **Random Forest** - An ensemble learning method using multiple decision trees
    3. **SVM (Support Vector Machine)** - Finds optimal hyperplane for classification
    4. **XGBoost** - Gradient boosting algorithm known for high performance
    
    ### 📈 Data Preprocessing Steps
    
    - Handling missing values (mode for categorical, median for numerical)
    - One-hot encoding for categorical variables (Sex, Race)
    - Label encoding for Marital status
    - SMOTE for handling class imbalance
    - Standard scaling for numerical features
    
    ### 🎯 How to Use
    
    1. **Data Analysis** - Explore the dataset, distributions, and correlations
    2. **Model Training** - Train models and compare performance
    3. **Single Prediction** - Enter patient data for individual risk assessment
    
    ---
    
    **Note:** This tool is for educational and research purposes. Always consult healthcare professionals for medical advice.
    """)

# Footer
st.markdown("---")
st.markdown("*Metabolic Syndrome Prediction System | Built with Streamlit | For Educational Purposes*")