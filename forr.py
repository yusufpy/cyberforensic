import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.preprocessing import LabelEncoder

# Load the saved model
def load_model():
    with open('voting_model.pkl', 'rb') as file:
        return pickle.load(file)

# Load Label Encoders
def load_label_encoders():
    encoders = {}
    for col in ['Activity_Type', 'Resource_Accessed', 'Anomaly_Type', 'Action']:
        with open(f'{col}_encoder.pkl', 'rb') as file:
            encoders[col] = pickle.load(file)
    return encoders

def generate_forensic_report(predicted_label, input_data):
    api_url = "https://api.openai.com/v1/chat/completions"  # Chat API for current models
    api_key = st.secrets["OPENAI_API_KEY"]   # Replace with your GPT API key

    # Prepare the messages for GPT
    messages = [
        {
            "role": "system",
            "content": "You are a cyber forensic expert generating concise reports."
        },
        {
            "role": "user",
            "content": f"""
            Cyber Forensic Report:
            The following prediction was made based on the input data:
            
            Predicted Label: {predicted_label}
            Input Data: {input_data.to_dict(orient='records')[0]}
            
            Generate a concise cyber forensic report detailing the potential threat, its context, and recommended actions.
            """
        }
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",  # Use `gpt-3.5-turbo` if `gpt-4` is unavailable
        "messages": messages,
        "max_tokens": 200
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Unable to generate report.")
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit app
def main():
    st.title("Cybercrime Forensic Prediction App")
    st.write("This app allows you to input feature values to make predictions using a pre-trained machine learning model and generate a cyber forensic report.")

    # Input features
    st.write("### Enter Feature Values:")
    hour = st.number_input("Hour (0-23):", min_value=0, max_value=23, value=12)
    day = st.number_input("Day (1-31):", min_value=1, max_value=31, value=15)
    day_of_week = st.number_input("Day of Week (0=Monday, 6=Sunday):", min_value=0, max_value=6, value=2)

    # Dropdown options
    activity_types = ['File_Modification', 'USB_insert', 'Network_Traffic']
    resources_accessed = ['/network/logs/new_project.docx', '/server/secrets.txt', '/project/document2.docx']
    anomaly_types = ['DDoS_Attempt', 'Brute_Force']
    actions = ['Read', 'Delete']

    # Dropdown inputs
    activity_type = st.selectbox("Activity Type:", options=activity_types)
    resource_accessed = st.selectbox("Resource Accessed:", options=resources_accessed)
    anomaly_type = st.selectbox("Anomaly Type:", options=anomaly_types)
    action = st.selectbox("Action:", options=actions)

    # Load label encoders
    encoders = load_label_encoders()

    # Encode categorical features
    try:
        activity_type_encoded = encoders['Activity_Type'].transform([activity_type])[0]
        resource_accessed_encoded = encoders['Resource_Accessed'].transform([resource_accessed])[0]
        anomaly_type_encoded = encoders['Anomaly_Type'].transform([anomaly_type])[0]
        action_encoded = encoders['Action'].transform([action])[0]
    except ValueError as e:
        st.error(f"Encoding error: {e}. Please ensure your input values are valid.")
        return

    # Prepare the input data
    input_data = pd.DataFrame({
        'Hour': [hour],
        'Day': [day],
        'DayOfWeek': [day_of_week],
        'Activity_Type': [activity_type_encoded],
        'Resource_Accessed': [resource_accessed_encoded],
        'Anomaly_Type': [anomaly_type_encoded],
        'Action': [action_encoded]
    })

    st.write("### Encoded Input Data:")
    st.dataframe(input_data)

    # Load model and make prediction
    model = load_model()
    if st.button("Predict and Generate Report"):
        prediction = model.predict(input_data)[0]
        st.write("### Prediction Result:")
        st.write(f"Predicted Label: {prediction}")

        # Generate forensic report
        st.write("### Cyber Forensic Report:")
        forensic_report = generate_forensic_report(prediction, input_data)
        st.text(forensic_report)

if __name__ == "__main__":
    main()
