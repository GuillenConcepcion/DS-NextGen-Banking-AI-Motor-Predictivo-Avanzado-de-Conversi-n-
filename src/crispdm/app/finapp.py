import streamlit as st
import pandas as pd
import mlflow
import logging

from crispdm.app.utils import generate_levels, generate_points, generate_radii_theta, create_parliament_chart, setup_layout
from crispdm.utils.config import settings

logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(layout="centered", page_title="Banking Conversion AI (MLOps v2.0)", page_icon="🏦")

@st.cache_resource
def load_model_from_mlflow():
    """Loads the model from MLflow using Aliases."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.model_name}@{settings.model_alias}"
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        st.error(f"Error loading model from MLflow. Ensure the tracking server is running and the model is registered. Error: {e}")
        st.stop()
        
    return model

# Input validation
def validate_inputs(inputs: dict) -> tuple[bool, str]:
    if inputs['day'] < 1 or inputs['day'] > 31:
        return False, "El día debe estar entre 1 y 31"
    if inputs['campaign'] < 1 or inputs['campaign'] > 50:
        return False, "El número de contactos debe estar entre 1 y 50"
    if inputs['pdays'] < -1 or inputs['pdays'] > 100:
        return False, "Los días desde el último contacto deben estar entre -1 y 100"
    return True, ""

# UI Layout
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("images/guillen_logo.png", width=120)
with col_title:
    st.title("Banking Conversion AI")
    st.markdown("#### *Enterprise MLOps Predictive Engine v2.0*")
    st.write("Enter customer information to predict the probability of subscribing to a term deposit.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    default = st.pills("Has credit in default?", ['yes', 'no'], default='no', key='default')
with col2:
    loan = st.pills("Has personal loan?", ['yes', 'no'], default='no', key='loan')
with col3:
    housing = st.pills("Has housing loan?", ['yes', 'no'], default='no', key='housing')
with col4:
    contact = st.pills("Contact Type", ['cellular', 'telephone'], default='cellular', key='contact')

month = st.pills("Month of last contact", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], default='may')
day = st.slider("Day of the month", 1, 31, 15)
campaign = st.slider("Contacts during this campaign", 1, 50, 1)
pdays = st.slider("Days since last contact from previous campaign", -1, 100, -1)

if st.button("Predict"):
    input_dict = {
        'default': default,
        'housing': housing,
        'loan': loan,
        'day': day,
        'contact': contact,
        'month': month,
        'campaign': campaign,
        'pdays': pdays
    }
    
    is_valid, message = validate_inputs(input_dict)
    if not is_valid:
        st.error(message, icon="⚠️")
        st.stop()
        
    input_data = pd.DataFrame([input_dict])
    
    try:
        # Lazy Loading: El modelo solo se carga (desde caché) cuando se presiona predecir
        model = load_model_from_mlflow()
        
        # Our PyFunc model handles all preprocessing steps!
        prediction_proba = model.predict(input_data)[0][1]
        prediction_percentage = round(prediction_proba * 100, 2)

        st.success(f"The predicted probability of the customer subscribing to a term deposit is: **{prediction_percentage}%**", icon="💻")

        # Visualizations
        parties = ['Not Convert', 'Probability of Conversion']
        probs = [(100 - int(prediction_percentage)) * 10, int(prediction_percentage) * 10]
        colors = ['#d3d3d3', '#57b45f']
        
        num_levels = 15
        levels = generate_levels(num_levels, min_radius=0.0, max_radius=1)
        total_probs = sum(probs)
        points_per_level = generate_points(levels, total_probs)
        radii_sorted, theta_sorted = generate_radii_theta(levels, points_per_level, 0, 180)
        
        fig = create_parliament_chart(parties, probs, colors, radii_sorted, theta_sorted, marker_size=6)
        
        title = "Conversion Probability"
        if prediction_percentage > 50:
            subtitle = f"This customer has a good chance of subscribing. Predicted probability: {prediction_percentage}%"
        else:
            subtitle = f"This customer has a low chance of subscribing. Predicted probability: {prediction_percentage}%"
            
        setup_layout(fig, title, subtitle)
        st.plotly_chart(fig, theme=None)
        
        # --- AI COPILOT SECTION ---
        st.divider()
        st.subheader("🤖 AI Sales Copilot")
        st.write("Recomendaciones tácticas para el agente de Call Center basadas en Inteligencia Artificial.")
        
        with st.spinner("Analizando el perfil y generando estrategia..."):
            from crispdm.app.ai_advisor import generate_sales_advice
            advice = generate_sales_advice(input_dict, prediction_proba)
            
            st.info(f"**Estrategia Sugerida ({advice['source']}):**\n\n{advice['strategy']}")
            st.success(f"**Guion Telefónico Recomendado:**\n\n*{advice['script']}*")
        st.divider()
        # --- END AI COPILOT SECTION ---
        
        # Explainable AI (SHAP)
        st.subheader("Model Interpretability (SHAP)")
        st.write("Understand *why* the model made this prediction.")
        
        with st.spinner("Generating explanation..."):
            import shap
            import matplotlib.pyplot as plt
            
            # Create a typical background client for the explainer
            background_df = pd.DataFrame([{
                'default': 'no', 'housing': 'no', 'loan': 'no',
                'day': 15, 'contact': 'cellular', 'month': 'may',
                'campaign': 1, 'pdays': -1
            }])
            
            def predict_fn(X):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=input_data.columns)
                return model.predict(X)[:, 1]
                
            explainer = shap.Explainer(predict_fn, background_df)
            shap_values = explainer(input_data)
            
            fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig_shap)
            
            # Prevent memory leaks in Streamlit
            fig_shap.clear()
            plt.close(fig_shap)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
