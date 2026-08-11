import os
import logging
from dotenv import load_dotenv

# Try to load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

def generate_sales_advice(customer_data: dict, prediction_prob: float) -> dict:
    """
    Generates a sales strategy and script based on the customer profile and prediction probability.
    Uses Google Gemini if GEMINI_API_KEY is present, otherwise falls back to a Smart Rule Engine.
    
    Returns:
        dict: Contains 'strategy' and 'script'.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if api_key and api_key != "tu_clave_api_gratuita_aqui":
        return _generate_with_gemini(customer_data, prediction_prob, api_key)
    else:
        return _generate_with_rules(customer_data, prediction_prob)

def _generate_with_gemini(customer_data: dict, prediction_prob: float, api_key: str) -> dict:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Eres un Copiloto Experto en Ventas de un banco. Tu objetivo es ayudar al operador del Call Center a vender un depósito a plazo fijo.
        
        Datos del cliente actual:
        - Préstamo personal: {customer_data.get('loan')}
        - Préstamo hipotecario: {customer_data.get('housing')}
        - Historial de impagos (default): {customer_data.get('default')}
        - Tipo de contacto: {customer_data.get('contact')}
        - Mes de contacto: {customer_data.get('month')}
        
        La Inteligencia Artificial predictiva estima que la probabilidad de que este cliente compre el producto es del {prediction_prob*100:.1f}%.
        
        Por favor, devuelve tu respuesta EXACTAMENTE en este formato (usa los delimitadores ###):
        
        ### ESTRATEGIA
        (Escribe aquí 2 líneas sobre cómo el agente debe abordar la llamada basado en los datos, por ejemplo: si tiene un préstamo, no presionarlo financieramente. Si la probabilidad es alta, usar cierre directo).
        
        ### GUION TELEFONICO
        (Escribe un guion corto de 3 líneas que el agente pueda leer, siendo empático y persuasivo).
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        text = response.text
        
        strategy = "Estrategia sugerida por la IA."
        script = "Hola, tenemos una oferta para ti."
        
        if "### ESTRATEGIA" in text and "### GUION" in text:
            parts = text.split("### GUION")
            strategy = parts[0].replace("### ESTRATEGIA", "").strip()
            script = parts[1].replace("TELEFONICO", "").strip()
            
        return {"strategy": strategy, "script": script, "source": "Gemini AI 🤖"}
        
    except Exception as e:
        logger.error(f"Gemini API failed: {e}. Falling back to Rule Engine.")
        return _generate_with_rules(customer_data, prediction_prob)

def _generate_with_rules(customer_data: dict, prediction_prob: float) -> dict:
    """Fallback rule-based engine if Gemini is not available or fails."""
    
    # Simple logic
    has_debts = customer_data.get('loan') == 'yes' or customer_data.get('housing') == 'yes'
    
    if prediction_prob >= 0.5:
        if has_debts:
            strategy = "Abordaje Consultivo: El cliente tiene alta probabilidad pero tiene deudas activas. Enfocar el depósito a plazo como una herramienta segura para generar retornos y ayudar a pagar sus deudas más rápido."
            script = "¡Hola! Sabemos que administrar deudas puede ser estresante. Hoy te llamo para ofrecerte un plan de ahorro a plazo fijo que te generará intereses garantizados, lo cual puede ayudarte a aliviar tus cuotas mensuales. ¿Te interesa saber más?"
        else:
            strategy = "Cierre Directo: El cliente tiene alta probabilidad y un perfil financiero limpio. Ofrecer los beneficios exclusivos y presionar suavemente para cerrar hoy."
            script = "¡Hola! Viendo tu excelente historial con nosotros, hemos pre-aprobado una tasa preferencial para nuestro depósito a plazo fijo. Es una oportunidad excelente para multiplicar tu dinero seguro. ¿Deseas que lo activemos ahora mismo?"
    else:
        if has_debts:
            strategy = "Abordaje Educativo (Baja Presión): Es poco probable que convierta debido a sus deudas. No presionar. Ofrecer educación financiera o un depósito mínimo."
            script = "Hola, me comunico del banco para presentarte brevemente una forma de empezar a ahorrar poco a poco con nuestro depósito a plazo, ideal para personas que buscan estabilizar sus finanzas. ¿Te enviamos la información al correo?"
        else:
            strategy = "Indagación de Necesidades: Perfil limpio pero baja probabilidad. Probablemente invierte en otro lado. Preguntar por sus metas financieras actuales."
            script = "Hola, noto que mantienes excelentes finanzas. Te llamo para contarte sobre nuestros depósitos a plazo. ¿Actualmente tienes alguna meta de ahorro o inversión en mente en la que podamos ayudarte a obtener mejores rendimientos?"
            
    return {"strategy": strategy, "script": script, "source": "Smart Rules ⚙️"}
