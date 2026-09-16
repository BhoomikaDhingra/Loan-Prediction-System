import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("best_pipeline.pkl")


try:
    model = load_model()

except Exception as e:
    st.error("❌ Could not load best_pipeline.pkl")
    st.error(
        "Make sure best_pipeline.pkl is in the same folder as app.py "
        "and that the scikit-learn version matches the model environment."
    )
    st.exception(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Loan Approval Prediction")

st.write(
    "Enter the applicant's details below to predict the "
    "loan approval outcome using the trained machine learning model."
)

st.divider()


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.subheader("👤 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=17,
        max_value=100,
        value=35,
        step=1
    )

with col2:
    job = st.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown"
        ]
    )

with col3:
    marital = st.selectbox(
        "Marital Status",
        [
            "divorced",
            "married",
            "single",
            "unknown"
        ]
    )


# ============================================================
# EDUCATION & CREDIT
# ============================================================

st.subheader("🎓 Education & Credit Information")

col1, col2, col3 = st.columns(3)

with col1:
    education = st.selectbox(
        "Education",
        [
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "illiterate",
            "professional.course",
            "university.degree",
            "unknown"
        ]
    )

with col2:
    default = st.selectbox(
        "Credit Default",
        [
            "no",
            "unknown",
            "yes"
        ]
    )

with col3:
    housing = st.selectbox(
        "Housing Loan",
        [
            "no",
            "unknown",
            "yes"
        ]
    )


# ============================================================
# LOAN INFORMATION
# ============================================================

st.subheader("💳 Loan Information")

loan = st.selectbox(
    "Personal Loan",
    [
        "no",
        "unknown",
        "yes"
    ]
)


# ============================================================
# CONTACT INFORMATION
# ============================================================

st.subheader("📞 Contact Information")

col1, col2, col3 = st.columns(3)

with col1:
    contact = st.selectbox(
        "Contact Type",
        [
            "cellular",
            "telephone"
        ]
    )

with col2:
    month = st.selectbox(
        "Contact Month",
        [
            "apr",
            "aug",
            "dec",
            "jul",
            "jun",
            "mar",
            "may",
            "nov",
            "oct",
            "sep"
        ]
    )

with col3:
    day_of_week = st.selectbox(
        "Day of Week",
        [
            "fri",
            "mon",
            "thu",
            "tue",
            "wed"
        ]
    )


# ============================================================
# CAMPAIGN INFORMATION
# ============================================================

st.subheader("📊 Campaign Information")

col1, col2 = st.columns(2)

with col1:
    campaign = st.number_input(
        "Number of Contacts in Current Campaign",
        min_value=1,
        max_value=56,
        value=2,
        step=1
    )

with col2:
    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        max_value=7,
        value=0,
        step=1
    )


# ============================================================
# PREVIOUS CONTACT INFORMATION
# ============================================================

st.subheader("📋 Previous Contact Information")

col1, col2 = st.columns(2)

with col1:

    if previous == 0:

        pdays = 999

        st.info(
            "No previous contacts selected → "
            "Days Since Previous Contact is automatically set to 999."
        )

    else:

        pdays = st.number_input(
            "Days Since Previous Contact",
            min_value=0,
            max_value=999,
            value=30,
            step=1
        )


with col2:

    if previous == 0:

        poutcome = "nonexistent"

        st.info(
            "No previous contacts selected → "
            "Previous Outcome is automatically set to nonexistent."
        )

    else:

        poutcome = st.selectbox(
            "Previous Outcome",
            [
                "failure",
                "nonexistent",
                "success"
            ]
        )


# ============================================================
# AUTOMATICALLY DERIVED FEATURES
# ============================================================

# No previous contact:
# If previous contacts = 0, automatically set to 1.
no_previous_contact = 1 if previous == 0 else 0


# Not working:
# Derived from the selected job.
non_working_jobs = [
    "retired",
    "student",
    "unemployed"
]

not_working = 1 if job in non_working_jobs else 0


# ============================================================
# CREATE THE 59 MODEL INPUT FEATURES
# ============================================================

input_data = {

    # Numerical features
    "age": age,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,

    # Automatically derived features
    "no_previous_contact": no_previous_contact,
    "not_working": not_working,

    # Job
    "job_admin.": 0,
    "job_blue-collar": 0,
    "job_entrepreneur": 0,
    "job_housemaid": 0,
    "job_management": 0,
    "job_retired": 0,
    "job_self-employed": 0,
    "job_services": 0,
    "job_student": 0,
    "job_technician": 0,
    "job_unemployed": 0,
    "job_unknown": 0,

    # Marital
    "marital_divorced": 0,
    "marital_married": 0,
    "marital_single": 0,
    "marital_unknown": 0,

    # Education
    "education_basic.4y": 0,
    "education_basic.6y": 0,
    "education_basic.9y": 0,
    "education_high.school": 0,
    "education_illiterate": 0,
    "education_professional.course": 0,
    "education_university.degree": 0,
    "education_unknown": 0,

    # Default
    "default_no": 0,
    "default_unknown": 0,
    "default_yes": 0,

    # Housing
    "housing_no": 0,
    "housing_unknown": 0,
    "housing_yes": 0,

    # Personal loan
    "loan_no": 0,
    "loan_unknown": 0,
    "loan_yes": 0,

    # Contact
    "contact_cellular": 0,
    "contact_telephone": 0,

    # Month
    "month_apr": 0,
    "month_aug": 0,
    "month_dec": 0,
    "month_jul": 0,
    "month_jun": 0,
    "month_mar": 0,
    "month_may": 0,
    "month_nov": 0,
    "month_oct": 0,
    "month_sep": 0,

    # Day of week
    "day_of_week_fri": 0,
    "day_of_week_mon": 0,
    "day_of_week_thu": 0,
    "day_of_week_tue": 0,
    "day_of_week_wed": 0,

    # Previous outcome
    "poutcome_failure": 0,
    "poutcome_nonexistent": 0,
    "poutcome_success": 0
}


# ============================================================
# SET SELECTED CATEGORIES TO 1
# ============================================================

input_data["job_" + job] = 1
input_data["marital_" + marital] = 1
input_data["education_" + education] = 1
input_data["default_" + default] = 1
input_data["housing_" + housing] = 1
input_data["loan_" + loan] = 1
input_data["contact_" + contact] = 1
input_data["month_" + month] = 1
input_data["day_of_week_" + day_of_week] = 1
input_data["poutcome_" + poutcome] = 1


# ============================================================
# CREATE DATAFRAME
# ============================================================

input_df = pd.DataFrame([input_data])


# ============================================================
# MATCH THE MODEL'S EXACT INPUT COLUMNS
# ============================================================

try:

    expected_columns = list(model.feature_names_in_)

except AttributeError:

    # Fallback in case feature_names_in_ is unavailable
    expected_columns = list(input_df.columns)


# Reorder columns exactly as the model expects.
input_df = input_df.reindex(
    columns=expected_columns,
    fill_value=0
)


# ============================================================
# VALIDATION
# ============================================================

if len(input_df.columns) != 59:

    st.error(
        f"Model input mismatch: expected 59 features, "
        f"but received {len(input_df.columns)}."
    )

    st.stop()


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "🔮 Check Loan Approval",
    use_container_width=True
):

    try:

        # Prediction
        prediction = model.predict(input_df)[0]

        # Probability
        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            # Find the probability corresponding to class 1
            classes = list(model.classes_)

            if 1 in classes:

                class_index = classes.index(1)
                probability = probabilities[class_index]


        # ====================================================
        # RESULT
        # ====================================================

        st.subheader("🏦 Loan Approval Result")

        if int(prediction) == 1:

            st.success("✅ PREDICTED OUTCOME: APPROVED")

            st.write(
                "Based on the information provided, "
                "the model predicts an approved outcome."
            )

        else:

            st.error("❌ PREDICTED OUTCOME: REJECTED")

            st.write(
                "Based on the information provided, "
                "the model predicts a rejected outcome."
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        if probability is not None:

            st.subheader("📊 Model Probability")

            st.metric(
                "Predicted Approval Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

            st.info(
                f"The model estimates a "
                f"{probability * 100:.2f}% probability "
                "for the approved class."
            )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.exception(e)


# ============================================================
# TECHNICAL INFORMATION
# ============================================================

with st.expander("🔧 Technical Information"):

    st.write(
        "Customer-facing inputs:",
        14
    )

    st.write(
        "Model input features:",
        len(input_df.columns)
    )

    st.write(
        "Input shape:",
        input_df.shape
    )

    st.write(
        "Expected model shape:",
        "(1, 59)"
    )