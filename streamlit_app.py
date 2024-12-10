import streamlit as st
import pandas as pd
import json
import boto3
import csv
import io
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# runtime = boto3.client(
#     "sagemaker-runtime",
#     region_name="us-east-1",
# )
IDENTITY_POOL_ID = "us-east-1:2ac8666d-0dab-4ad1-8584-fb59e6d5da4c"
ENDPOINT_NAME = "xgboost-2024-12-09-23-51-46-022"

def get_cognito_credentials(identity_pool_id, region_name="us-east-1"):
    try:
        # Initialize Cognito Identity client
        cognito = boto3.client("cognito-identity", region_name=region_name)
        
        # Get Identity ID
        identity_id = cognito.get_id(IdentityPoolId=identity_pool_id)["IdentityId"]
        
        # Get temporary credentials
        credentials = cognito.get_credentials_for_identity(IdentityId=identity_id)["Credentials"]
        return credentials
    except Exception as e:
        raise ValueError(f"Error retrieving Cognito credentials: {e}")

credentials = get_cognito_credentials(identity_pool_id=IDENTITY_POOL_ID)

runtime = boto3.client(
            "sagemaker-runtime",
            region_name="us-east-1",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretKey"],
            aws_session_token=credentials["SessionToken"],
        )

# Load scaler from training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def preprocess_input(input_data, scaler):
    try:
        # Convert the input dictionary to a pandas DataFrame
        df = pd.DataFrame([input_data])

        # Ensure all columns are treated as numeric for scaling
        numeric_cols = df.columns

        # Scale the DataFrame
        scaled_df = pd.DataFrame(scaler.transform(df[numeric_cols]), columns=numeric_cols)

        # Convert the scaled DataFrame back to a dictionary
        scaled_data = scaled_df.iloc[0].to_dict()

        return scaled_data
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

def convert_dict_to_csv(input_data):
    """
    Converts a dictionary of input data into a CSV string format without headers.

    Parameters:
        input_data (dict): The input features for the model.

    Returns:
        str: A CSV-formatted string.
    """
    try:
        # Extract values from the dictionary
        csv_data = ",".join(map(str, input_data.values()))
        return csv_data
    except Exception as e:
        raise ValueError(f"Failed to convert input_data to CSV: {e}")


def predict_sagemaker(input_data, endpoint_name):
    """
    Sends input data to the specified SageMaker endpoint and returns the prediction.

    Parameters:
        input_data (dict): The input features for the model.
        endpoint_name (str): The name of the deployed SageMaker endpoint.

    Returns:
        dict: A dictionary with the prediction or an error message.
    """
    try:
        # Convert the input data dictionary to CSV format
        input_csv = convert_dict_to_csv(input_data)
        
        # Send the request to the SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=input_csv,
        )
        
        # Read and decode the response
        prediction = response["Body"].read().decode("utf-8").strip()
        
        # Split the predictions (assuming comma-separated values in the response)
        predictions_array = np.fromstring(prediction, sep=',')
        
        # Round the predictions
        rounded_predictions = np.round(predictions_array)

        # Convert the rounded predictions into risk messages
        if rounded_predictions[0] == 0:
            risk_message = (
                "We are pleased to inform you that our predictive model indicates a low likelihood of stillbirth based on the "
                "information you provided. While this is encouraging, ongoing prenatal care remains essential for ensuring a healthy pregnancy.\n\n"
                "**Recommendations for Continued Care:**\n"
                "- **Attend Regular Check-ups**: Keep all scheduled appointments to monitor your pregnancy.\n"
                "- **Monitor Signs and Symptoms**: Stay attentive to your body and your baby’s movements, and report any concerns to your healthcare provider.\n"
                "- **Maintain a Healthy Lifestyle**: Follow medical advice on nutrition, exercise, and stress management.\n\n"
                "For additional information, visit our *Tips for Success* section on our website. It provides valuable insights "
                "to help you maintain a healthy pregnancy.\n\n"
                "Thank you for your commitment to your health and your baby’s well-being. If you have any concerns, please reach out to your "
                "healthcare provider. Wishing you a smooth and healthy pregnancy!"
            )
        else:
            risk_message = (
                "We regret to inform you that our model has identified a potential high risk for stillbirth based on the "
                "information you provided. This is not a guarantee of stillbirth but an indication that further medical "
                "evaluation is crucial. We strongly recommend scheduling an appointment with your healthcare provider "
                "immediately to discuss these results and determine the best course of action.\n\n"
                "**Immediate Steps to Take:**\n"
                "- **Consult a Healthcare Provider**: Schedule an appointment as soon as possible.\n"
                "- **Monitor Symptoms**: Pay close attention to changes in symptoms or fetal movements and report them promptly.\n"
                "- **Seek Support**: Reach out to loved ones or support groups during this challenging time.\n"
                "- **Maintain a Healthy Lifestyle**: Focus on a balanced diet, appropriate physical activity, and stress management.\n\n"
                "For additional guidance, please visit our *Tips for Success* section on our website, where you’ll find "
                "helpful strategies and resources.\n\n"
                "Your health and your baby’s well-being are our utmost priority. With timely intervention, the risk can often be mitigated. "
                "Wishing you strength and support during this time."
            )
        # Return a dictionary with success status and risk message
        return risk_message
    except Exception as e:
        return {"error": str(e)}

# Set page configuration
st.set_page_config(page_title="StillSafe", page_icon="🤰", layout="wide")

# Inject custom CSS for styling
st.markdown(
    """
    <style>
        /* Sidebar background and text styling */
        div[data-testid="stSidebar"] {
            background-color: #FEEBF3; /* Soft pink */
        }
        div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3,
        div[data-testid="stSidebar"] h4, div[data-testid="stSidebar"] h5, div[data-testid="stSidebar"] h6,
        div[data-testid="stSidebar"] p, div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] span {
            color: #3D405B; /* Warm gray for text */
        }
        /* Main content area background and text styling */
        div[data-testid="stAppViewContainer"] {
            background-color: #FFF5F8; /* Light pastel pink */
            color: #3D405B; /* Warm gray for main content */
        }
        div[data-testid="stAppViewContainer"] h1,
        div[data-testid="stAppViewContainer"] h2,
        div[data-testid="stAppViewContainer"] h3,
        div[data-testid="stAppViewContainer"] h4,
        div[data-testid="stAppViewContainer"] h5,
        div[data-testid="stAppViewContainer"] h6 {
            color: #C45BAA !important; /* Feminine magenta for headings */
        }
        /* Input label (question styling) */
        div[data-testid="stAppViewContainer"] label {
            color: #3D405B !important; /* Match greyish color */
            font-weight: bold; /* Bold labels */
        }
        /* Dropdown and input text styling */
        select, input, textarea {
            color: white !important; /* White text inside input fields */
            background-color: #3D405B !important; /* Dark gray background for input fields */
        }
        option {
            color: white !important; /* White text for dropdown options */
        }
        textarea {
            color: white !important; /* White text for text area */
        }
        /* Button styling */
        .stButton button {
            background-color: #FFC8E1; /* Light pink */
            color: #3D405B; /* Warm gray text */
            border-radius: 12px;
            border: 1px solid #C45BAA; /* Magenta border */
            padding: 8px 16px;
        }
        .stButton button:hover {
            background-color: #FEEBF3; /* Softer pink on hover */
            color: #3D405B; /* Warm gray text */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for tab navigation
if "tab_selection" not in st.session_state:
    st.session_state["tab_selection"] = "Home"  # Default to Home tab

# Sidebar for Navigation
st.sidebar.title("Navigation")
tab_selection = st.sidebar.radio(
    "Go to",
    ["Home", "Risk Assessment", "Meet Our Team","StillSafe Tips for Success", "Feedback"],
    index=["Home", "Risk Assessment", "Meet Our Team","StillSafe Tips for Success", "Feedback"].index(st.session_state["tab_selection"])
)

# Update session state with sidebar selection
if tab_selection != st.session_state["tab_selection"]:
    st.session_state["tab_selection"] = tab_selection

# Helper functions
def convert_month_to_number(month_name):
    months = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    return months.get(month_name)

def convert_race_to_code(race_name):
    races = [
        "White (alone)", "Black (alone)", "AIAN (alone)", "Asian (alone)", "NHOPI (alone)",
        "Black And White", "Black and AIAN", "Black and Asian", "Black and NHOPI",
        "AIAN and White", "AIAN and Asian", "AIAN and NHOPI", "Asian and White",
        "Asian and NHOPI", "NHOPI and White", "Black, AIAN, and White", "Black, AIAN, and Asian",
        "Black, AIAN, and NHOPI", "Black, Asian, and White", "Black, Asian, and NHOPI",
        "Black, NHOPI, and White", "AIAN, Asian, and White", "AIAN, NHOPI, and White",
        "AIAN, Asian, and NHOPI", "Asian, NHOPI, and White", "Black, AIAN, Asian, and White",
        "Black, AIAN, Asian, and NHOPI", "Black, AIAN, NHOPI, and White",
        "Black, Asian, NHOPI, and White", "AIAN, Asian, NHOPI, and White",
        "Black, AIAN, Asian, NHOPI, and White"
    ]
    return races.index(race_name) + 1

def convert_education_to_code(education_level):
    education_levels = [
        "8th grade or less", "9th through 12th grade with no diploma",
        "High school graduate or GED completed", "Some college credit, but not a degree",
        "Associate degree (AA, AS)", "Bachelor’s degree (BA, AB, BS)",
        "Master’s degree (MA, MS, MEng, Med, MSW, MBA)",
        "Doctorate (PhD, EdD) or Professional degree (MD, DDS, DVM, LLB, JD)"
    ]
    return education_levels.index(education_level) + 1

def calculate_bmi(weight_pounds, height_inches):
    if height_inches > 0:  # Prevent division by zero
        return (weight_pounds / (height_inches ** 2)) * 703
    return None

def convert_sex_to_binary(sex):
    return 1 if sex == "Male" else 0

def process_last_birth_months(months):
    return {
        "Last_Birth_Less_than_1_year": int(months < 12),
        "Last_Birth_1_year_to_2_5_years": int(12 <= months < 30),
        "Last_Birth_2_5_years_to_4_years": int(30 <= months < 48),
        "Last_Birth_4_to_5_5_years": int(48 <= months < 66),
        "Last_Birth_Greater_than_5_5_years": int(months >= 66)
    }

def convert_yes_no_to_binary(response):
    return 1 if response == "Yes" else 0

def determine_previous_birth(total_prior_births):
    return 1 if total_prior_births > 0 else 0


# Home Tab
if st.session_state["tab_selection"] == "Home":
    # Centered Logo at the Top
    col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns for centering
    with col2:
        st.image("images/finallogo.jpg", caption=None, use_container_width=False, width=350)  # Adjust width to 150px

    # About StillSafe
    st.markdown("<h3 style='color:#C45BAA;'>About StillSafe</h3>", unsafe_allow_html=True)
    st.write("""
    At **StillSafe**, we are on a mission to reduce the heartbreak of stillbirth by providing expecting families with the tools, 
    knowledge, and support they need for a safe pregnancy journey. By harnessing the power of advanced machine learning, we aim to 
    identify risks early—empowering families with personalized insights to make informed decisions.
    """)

    # Why We Care
    st.markdown("<h3 style='color:#C45BAA;'>Why We Care</h3>", unsafe_allow_html=True)
    st.write("""
    Each year, approximately 21,000 families in the United States endure the devastating loss of a stillbirth—a tragedy 
    that is often preventable. Research shows that up to 80% of stillbirths may be avoided with early intervention. 
    At **StillSafe**, we turn data into action, creating innovative tools to help families and healthcare providers 
    collaborate for safer pregnancies.
    """)

    # Our Mission
    st.markdown("<h3 style='color:#C45BAA;'>Our Mission</h3>", unsafe_allow_html=True)
    st.write("""
    To ensure no family experiences a preventable stillbirth by providing accessible, evidence-based 
    risk assessment tools and empowering support throughout pregnancy.
    """)

    # How It Works
    st.markdown("<h3 style='color:#C45BAA;'>How It Works</h3>", unsafe_allow_html=True)
    st.write("""
    - **Input Key Information:** Enter basic health and demographic details into our secure, user-friendly platform.
    - **Personalized Risk Assessment:** Our advanced machine learning tool identifies pregnancies at higher risk for stillbirth.
    - **Tips for Success:** Explore our medically-approved Guide to a Safe and Healthy Pregnancy for actionable recommendations.
    - **Feedback Section:** Share your thoughts and help us improve by providing feedback directly through our platform.
    """)

    # Image and Content Layout
    col1, col2 = st.columns([1, 2])  # Adjust column ratios as needed

    # Left Column: Image
    with col1:
        st.image("images/pregnancypic.jpg", caption="Supporting your pregnancy journey.", use_container_width=True)

    # Right Column: Center the content vertically
    with col2:
        # Add vertical spacing to align content to the middle of the image
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)  # Adjust height as needed

        # Centered Heading
        st.markdown("<h3 style='text-align:left;color:#C45BAA;'>Get Started Today</h3>", unsafe_allow_html=True)

        # Centered Text
        st.markdown("""
        <p style="text-align: left; font-size: 16px;">
         Take the first step toward a safer pregnancy. Whether you’re planning, expecting, or supporting someone who is, 
         <strong>StillSafe</strong> is here to help.
        </p>
        """, unsafe_allow_html=True)

        # Button to navigate to Risk Assessment
        if st.button("Start Your Risk Assessment Now", key="home_risk_button"):
            st.session_state["tab_selection"] = "Risk Assessment"

# Risk Assessment Tab
elif st.session_state["tab_selection"] == "Risk Assessment":
    # Create a two-column layout for the logo and the title
    col1, col2 = st.columns([1, 5])  # Adjust column widths as needed

    # Logo in the left column
    with col1:
        st.image("images/finallogoonly.jpg", width=200)  # Adjust the width as necessary

    # Title in the right column
    with col2:
        st.markdown("<h1 style='color:#C45BAA;'>Risk Assessment Tool</h1>", unsafe_allow_html=True)

    st.write("Provide the following information to receive a personalized pregnancy risk assessment.")

    # User Inputs
    delivery_month = st.selectbox("What month are you expecting to have your baby?", 
                                ["January", "February", "March", "April", "May", "June",
                                    "July", "August", "September", "October", "November", "December"])
    delivery_month_num = convert_month_to_number(delivery_month)

    mothers_age = st.slider("How old are you right now?", 0, 65, 30)

    mothers_race = st.selectbox("What is your race?", [
        "White (alone)", "Black (alone)", "AIAN (alone)", "Asian (alone)", "NHOPI (alone)",
        "Black And White", "Black and AIAN", "Black and Asian", "Black and NHOPI",
        "AIAN and White", "AIAN and Asian", "AIAN and NHOPI", "Asian and White",
        "Asian and NHOPI", "NHOPI and White", "Black, AIAN, and White", "Black, AIAN, and Asian",
        "Black, AIAN, and NHOPI", "Black, Asian, and White", "Black, Asian, and NHOPI",
        "Black, NHOPI, and White", "AIAN, Asian, and White", "AIAN, NHOPI, and White",
        "AIAN, Asian, and NHOPI", "Asian, NHOPI, and White", "Black, AIAN, Asian, and White",
        "Black, AIAN, Asian, and NHOPI", "Black, AIAN, NHOPI, and White",
        "Black, Asian, NHOPI, and White", "AIAN, Asian, NHOPI, and White",
        "Black, AIAN, Asian, NHOPI, and White"
    ])
    mothers_race_code = convert_race_to_code(mothers_race)

    mothers_education = st.selectbox("What is your highest level of education?", [
        "8th grade or less", "9th through 12th grade with no diploma",
        "High school graduate or GED completed", "Some college credit, but not a degree",
        "Associate degree (AA, AS)", "Bachelor’s degree (BA, AB, BS)",
        "Master’s degree (MA, MS, MEng, Med, MSW, MBA)",
        "Doctorate (PhD, EdD) or Professional degree (MD, DDS, DVM, LLB, JD)"
    ])
    mothers_education_code = convert_education_to_code(mothers_education)

    fathers_age = st.slider("How old is the baby's father?", 0, 100, 30)

    prenatal_care_month = st.slider("How many months along in your pregnancy were you when you had your first prenatal care visit? If you have yet to visit, at how many months do you expect to have your first prenatal care visit?", 1, 10)

    # Prepregnancy Weight and Height Inputs (Side by Side)
    col1, col2 = st.columns(2)
    with col1:
        weight_pounds = st.number_input("What was your pre-pregnancy weight in pounds?", min_value=50.0, max_value=500.0, step=0.1, value=120.0)
    with col2:
        height_inches = st.number_input("What was your pre-pregnancy height in inches?", min_value=48.0, max_value=96.0, step=0.1, value=60.0)

    # Calculate BMI
    mothers_bmi = calculate_bmi(weight_pounds, height_inches)
    if mothers_bmi:
        st.write(f"Your calculated pre-pregnancy BMI: {mothers_bmi:.2f}")

    diabetes_prepregnancy = st.selectbox("Did you have diabetes pre-pregnancy?", ["No", "Yes"])
    diabetes_prepregnancy_binary = convert_yes_no_to_binary(diabetes_prepregnancy)

    gestational_diabetes = st.selectbox("Did you get gestational diabetes?", ["No", "Yes"])
    gestational_diabetes_binary = convert_yes_no_to_binary(gestational_diabetes)

    prep_hypertension = st.selectbox("Did you have pre-pregnancy hypertension?", ["No", "Yes"])
    prep_hypertension_binary = convert_yes_no_to_binary(prep_hypertension)

    gestational_hypertension = st.selectbox("Did you get gestational hypertension?", ["No", "Yes"])
    gestational_hypertension_binary = convert_yes_no_to_binary(gestational_hypertension)

    hypertension_eclampsia = st.selectbox("Do you have hypertension eclampsia?", ["No", "Yes"])
    hypertension_eclampsia_binary = convert_yes_no_to_binary(hypertension_eclampsia)

    infertility_treatment = st.selectbox("Have you undergone infertility treatment?", ["No", "Yes"])
    infertility_treatment_binary = convert_yes_no_to_binary(infertility_treatment)

    infant_sex = st.selectbox("What is the expected sex of your baby?", ["Male", "Female"])
    infant_sex_binary = convert_sex_to_binary(infant_sex)

    wic_program = st.selectbox("Are you participating in the WIC program? (Supplemental Nutrition Assistance)", ["No", "Yes"])
    wic_program_binary = convert_yes_no_to_binary(wic_program)

    cigarettes_during_pregnancy = st.selectbox("Have you been smoking cigarettes during your pregnancy?", ["No", "Yes"])
    cigarettes_during_pregnancy_binary = convert_yes_no_to_binary(cigarettes_during_pregnancy)

    cigarettes_before_pregnancy = st.selectbox("Did you smoke cigarettes before your pregnancy?", ["No", "Yes"])
    cigarettes_before_pregnancy_binary = convert_yes_no_to_binary(cigarettes_before_pregnancy)

    total_prior_births = st.slider("How many previous pregnancies have you had?", 0, 21, 0)

    had_previous_birth_binary = determine_previous_birth(total_prior_births)

    last_birth_months = st.slider("How many months has it been since your last pregnancy? Put 0 if you have not had any previous pregnancies.", 0, 320, 0)
    last_birth_features = process_last_birth_months(last_birth_months)

    risk_sum = (
    diabetes_prepregnancy_binary +
    gestational_diabetes_binary +
    prep_hypertension_binary +
    gestational_hypertension_binary +
    hypertension_eclampsia_binary +
    infertility_treatment_binary +
    cigarettes_during_pregnancy_binary +
    cigarettes_before_pregnancy_binary
)
        # Add logic to process inputs
    if st.button("Submit"):
        # Input data dictionary
        input_data = {
            "Delivery_Month": delivery_month_num,
            "Mothers_Age": mothers_age,
            "Mothers_Race_Recode_31": mothers_race_code,
            "Mothers_Education": mothers_education_code,
            "Fathers_Age_Combined": fathers_age,
            "Month_Prenatal_Care_Began": prenatal_care_month,
            "Mothers_PrePregnancy_BMI": mothers_bmi,
            "Diabetes_Prepregnancy": diabetes_prepregnancy_binary,
            "Gestational_Diabetes": gestational_diabetes_binary,
            "PrePregnancy_Hypertension": prep_hypertension_binary,
            "Gestational_Hypertension": gestational_hypertension_binary,
            "Hypertension_Eclampsia": hypertension_eclampsia_binary,
            "Infertility_Treatment": infertility_treatment_binary,
            "Infant_Sex": infant_sex_binary,
            "WIC_Program": wic_program_binary,
            "Cigarettes_During_Pregnancy": cigarettes_during_pregnancy_binary,
            "Cigarettes_Before_Pregnancy_Int": cigarettes_before_pregnancy_binary,
            "Total_Prior_Births": total_prior_births,
            "Had_Previous_Birth": had_previous_birth_binary,
            "Less_than_1_year": last_birth_features["Last_Birth_Less_than_1_year"],
            "1_year_to_2.5_years": last_birth_features["Last_Birth_1_year_to_2_5_years"],
            "2.5_years_to_4_years": last_birth_features["Last_Birth_2_5_years_to_4_years"],
            "4_to_5.5_years": last_birth_features["Last_Birth_4_to_5_5_years"],
            "Greater_than_5.5_years": last_birth_features["Last_Birth_Greater_than_5_5_years"],
            "Risk_Sum": risk_sum
        }

        # Preprocess input data
        preprocessed_data = preprocess_input(input_data, scaler)

        # Send preprocessed data to SageMaker
        prediction = predict_sagemaker(preprocessed_data, ENDPOINT_NAME)

        # Display prediction
        st.markdown(
            f"""
            <div style="background-color: #EAFBF1; border-left: 5px solid #62A87C; padding: 10px; border-radius: 10px; margin-top: 20px;">
                <p style="color: #3D405B; font-size: 16px; margin: 0;">
                    <strong>Risk Assessment: </strong>{prediction}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # st.markdown(
        #     f"""
        #     <div style="background-color: #EAFBF1; border-left: 5px solid #62A87C; padding: 10px; border-radius: 10px; margin-top: 20px;">
        #         <p style="color: #3D405B; font-size: 16px; margin: 0;">
        #             <strong>Prediction:</strong> {prediction}
        #         </p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )

    st.markdown("---")
    st.markdown("<h3 style='color:#C45BAA;'>Disclaimer</h3>", unsafe_allow_html=True)
    st.write("""
    The **StillSafe Risk Assessment Tool** is designed by data science students and professionals to offer data-driven insights. 
    However, it is not intended to replace medical advice, diagnosis, or treatment. For any concerns about your pregnancy or health, always consult a qualified healthcare provider. 
    Your health and well-being are our top priority.
    """)

# About Us Tab
elif st.session_state["tab_selection"] == "Meet Our Team":
    # Create a two-column layout for the logo and the title
    col1, col2 = st.columns([1, 5])  # Adjust column widths as needed

    # Logo in the left column
    with col1:
        st.image("images/finallogoonly.jpg", width=200)  # Adjust the width as necessary

    # Title in the right column
    with col2:
        st.markdown("<h1 style='color:#C45BAA;'>Meet Our Team</h1>", unsafe_allow_html=True)
    st.write("At StillSafe, we’re a team of Master’s students from UC Berkeley, studying Information and Data Science, brought together by a shared passion for supporting moms-to-be and their little ones. We’re dedicated to using technology in a thoughtful, caring way to create tools that truly make a difference. With our hearts in the right place and our skills at work, we’re here to help make every pregnancy journey safer, more supported, and filled with confidence.")
    
    # Team Members with Photos and Emails
    team_members = [
        {"name": "Jonah Grossman", "role": "Project Manager, Head Developer, MLE & Master in Information and Data Science at UC Berkeley", "email": "jonahgrossman0@berkeley.edu", "photo": "images/jonahfinal.jpg"},
        {"name": "Joshua Shin", "role": "MLE & Master in Information and Data Science at UC Berkeley", "email": "joony9191@berkeley.edu", "photo": "images/joshfinal.jpg"},
        {"name": "Millie Kobayashi", "role": "Data Scientist, MLE & Master in Information and Data Science at UC Berkeley", "email": "milliek@berkeley.edu", "photo": "images/milliefinal.jpg"},
        {"name": "Kelechi Nnebedum", "role": "Data Scientist & Master in Information and Data Science at UC Berkeley", "email": "knnebedum@berkeley.edu", "photo": "images/kelechifinal.jpg"},
        {"name": "Adithi Suresh", "role": "Developer, Designer & Master in Information and Data Science at UC Berkeley", "email": "adithi_suresh@berkeley.edu", "photo": "images/adithifinal.jpg"},
        {"name": "Nikita Chauhan", "role": "Data Scientist, Designer & Master in Information and Data Science at UC Berkeley", "email": "nikitac@berkeley.edu", "photo": "images/nikitafinal.jpg"},
    ]

    # Display each team member with photo, role, and email
    for member in team_members:
        col1, col2 = st.columns([1, 4])  # Define column layout
        with col1:
            try:
                st.image(member["photo"], use_container_width=True)  # Load the image
            except Exception as e:
                st.warning(f"Could not load image for {member['name']}. Make sure the file exists at '{member['photo']}'.")
        with col2:
            st.markdown(f"### {member['name']}")
            st.markdown(f"**{member['role']}**")
            st.markdown(f"📧 **Email:** [{member['email']}](mailto:{member['email']})")

    st.markdown("---")
    st.markdown("<h3 style='color:#C45BAA;'>Contact Us</h3>", unsafe_allow_html=True)
    st.write("""
    **Have a question, concern, or just curious to learn more? We’re here to help! Reach out to our friendly team anytime – we’d love to hear from you.**

    📧 **Email:** jonahgrossman0@berkeley.edu  

    📞 **Phone:** (818) 312-3752  
""")

#StillSafe Tips for Success Tab
elif st.session_state["tab_selection"] == "StillSafe Tips for Success":
    # Create a two-column layout for the logo and the title
    col1, col2 = st.columns([1, 5])  # Adjust column widths as needed

    # Logo in the left column
    with col1:
        st.image("images/finallogoonly.jpg", width=200)  # Adjust the width as necessary

    # Title in the right column
    with col2:
        st.markdown("<h1 style='color:#C45BAA;'>StillSafe Tips for Success</h1>", unsafe_allow_html=True)
    st.title("StillSafe: Your Guide to a Safe and Healthy Pregnancy")

    # Tip 1: Prenatal Checkups
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/checkupsfinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Schedule Regular Prenatal Checkups</h3>
            Work closely with your healthcare provider to ensure everything is on track and to address any questions or concerns along the way. 
            They’re your trusted partner in this journey, so never hesitate to share how you’re feeling—they’re here to support both you and your baby!
        """, unsafe_allow_html=True)

    # Tip 2: Balanced Diet
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/dietfinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Nourish Your Body with a Balanced Diet</h3>
            Eating well during pregnancy is one of the best gifts you can give yourself and your baby. 
            Focus on nutrient-rich foods, including fresh fruits, vegetables, whole grains, lean proteins, and dairy. 
            Treat yourself to wholesome, nourishing meals—it’s a small step with a big impact on you and your little one’s health!
        """, unsafe_allow_html=True)

    # Tip 3: Physical Activity
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/exercisefinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Engage in Safe Physical Activity</h3>
            Staying active during pregnancy can boost your energy, improve your mood, and promote better sleep. 
            Try gentle exercises like walking, prenatal yoga, swimming, or low-impact aerobics—just make sure to get the green light from your healthcare provider. 
            These activities not only support your physical health but also help you stay mentally balanced and prepared for the journey ahead!
        """, unsafe_allow_html=True)

    # Tip 4: Avoid Harmful Substances
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/avoid_harmfulfinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Avoid Harmful Substances</h3>
            Keeping your baby’s development safe and sound starts with making healthy choices for yourself. 
            Try to minimize caffeine, and be sure to steer clear of alcohol, tobacco, and recreational drugs to give your little one the best possible start. 
            Remember, every small step you take toward a healthier lifestyle is a big step for your baby’s well-being!
        """, unsafe_allow_html=True)

    # Tip 5: Stay Hydrated
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/hydrationfinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Stay Hydrated</h3>
            Keeping hydrated is one of the simplest and most effective ways to care for yourself and your baby during pregnancy. 
            Aim to drink plenty of water throughout the day to support your body’s increased demands, maintain healthy circulation, 
            and reduce the risk of common pregnancy discomforts like swelling and constipation. 
            Carry a water bottle with you as a reminder, and consider adding slices of lemon, cucumber, or fresh fruit for a refreshing twist!
        """, unsafe_allow_html=True)

    # Tip 6: Rest and Relaxation
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/relaxationfinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Prioritize Rest and Relaxation</h3>
            Your body is working hard to support your baby, so getting 7-9 hours of quality sleep each night is essential. 
            Create a cozy bedtime routine, and try relaxation techniques like mindfulness, light stretching, or a calming cup of herbal tea to unwind. 
            Resting well helps you recharge and prepare for the exciting journey ahead!
        """, unsafe_allow_html=True)

    # Tip 7: Empower with Knowledge
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/knowledgefinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Empower Yourself with Knowledge</h3>
            Knowledge is power, especially when it comes to your pregnancy journey! 
            Explore trusted resources for accurate, up-to-date information, and don’t hesitate to ask your healthcare provider any questions you have—they’re there to guide you. 
            The more you know, the more confident and prepared you’ll feel as you await your baby’s arrival!
        """, unsafe_allow_html=True)

    # Tip 8: Support System
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/supportfinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Create Your Circle of Support</h3>
            Surround yourself with loving family, friends, or a community of other parents-to-be. 
            Having a strong support network can uplift your spirits and offer the encouragement you need as you prepare for your baby’s arrival.
        """, unsafe_allow_html=True)

    # Tip 9: Baby’s Big Day
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/baby_dayfinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Prepare for Your Baby’s Big Day</h3>
            Get ready for your little one’s arrival by chatting with your healthcare provider about your birth plan. 
            Take some time to explore your hospital or birthing center's procedures, and don’t forget to pack a bag with all the essentials for labor and postpartum recovery—it’s one step closer to meeting your baby!
        """, unsafe_allow_html=True)

    # Tip 10: Monitor Your Health
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/health_monitoringfinal.jpg", use_container_width=True)  # Updated parameter
    with col2:
        st.markdown("""
            <h3 style='color:#C45BAA;'>Keep an Eye on Your Health</h3>
            Listen to your body and reach out to your healthcare provider if you notice anything unusual, like persistent headaches, swelling, or changes in your baby’s movements. 
            Catching concerns early can help ensure a smoother, healthier journey for you and your little one.
        """, unsafe_allow_html=True)

# Feedback Tab
elif st.session_state["tab_selection"] == "Feedback":
    # Create a two-column layout for the logo and the title
    col1, col2 = st.columns([1, 5])  # Adjust column widths as needed

    # Logo in the left column
    with col1:
        st.image("images/finallogoonly.jpg", width=200)  # Adjust the width as necessary

    # Title in the right column
    with col2:
        st.markdown("<h1 style='color:#C45BAA;'>Feedback</h1>", unsafe_allow_html=True)
    st.title("We’d Love to Hear From You!")
    st.write("Welcome to our feedback section! Please feel free to ask us any questions or share your feedback.")
    
    feedback = st.text_area("Your Feedback", placeholder="Type your message here...")
    
    if st.button("Submit Feedback"):
        if feedback.strip():
            
# Custom-styled success message
            st.markdown(
                """
                <div style="background-color: #EAFBF1; border-left: 5px solid #62A87C; padding: 10px; border-radius: 10px; margin-top: 20px;">
                    <p style="color: #3D405B; font-size: 16px; margin: 0;">
                        <strong>Thank you for your feedback!</strong> We’ll get back to you shortly.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Custom-styled error message
            st.markdown(
                """
                <div style="background-color: #FDEDEC; border-left: 5px solid #E74C3C; padding: 10px; border-radius: 10px; margin-top: 20px;">
                    <p style="color: #3D405B; font-size: 16px; margin: 0;">
                        <strong>Please enter some feedback</strong> before submitting.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )