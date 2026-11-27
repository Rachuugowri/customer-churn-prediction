
import os
import sqlite3
import hashlib
import secrets
import time

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# 2. FILE PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "users.db")

DATASET_PATH = os.path.join(
    BASE_DIR,
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# =====================================================
# 3. DARK THEME
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #080d1c;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111a32;
}

h1, h2, h3 {
    color: white;
}

p, label {
    color: #d0d7e8;
}

.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    color: white;
    background: linear-gradient(90deg, #6d28d9, #0891b2);
}

div[data-testid="stMetric"] {
    background-color: #151f3b;
    border: 1px solid #29375c;
    padding: 20px;
    border-radius: 12px;
}

.card {
    background: #151f3b;
    border: 1px solid #29375c;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. DATABASE CREATION AND MIGRATION
# =====================================================

def create_database():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name='users'
    """)

    table_exists = cursor.fetchone()

    if table_exists is None:

        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT,
                password_hash TEXT,
                salt TEXT
            )
        """)

    else:

        cursor.execute("PRAGMA table_info(users)")

        column_info = cursor.fetchall()
        columns = [row[1] for row in column_info]

        if "username" not in columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN username TEXT"
            )

        if "email" not in columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN email TEXT"
            )

        if "password" not in columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN password TEXT"
            )

        if "password_hash" not in columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN password_hash TEXT"
            )

        if "salt" not in columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN salt TEXT"
            )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            email TEXT PRIMARY KEY,
            reset_code_hash TEXT NOT NULL,
            expires_at REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()


create_database()


# =====================================================
# 5. PASSWORD HASHING
# =====================================================

def hash_password(password, salt):

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000
    )

    return hashed.hex()


# =====================================================
# 6. REGISTRATION
# =====================================================

def register_user(username, email, password):

    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "Please fill all fields."

    if len(password) < 6:
        return False, "Password must contain at least 6 characters."

    salt = secrets.token_hex(16)

    password_hash = hash_password(password, salt)

    conn = sqlite3.connect(DB_PATH)

    try:

        conn.execute("""
            INSERT INTO users
            (username, email, password, password_hash, salt)
            VALUES (?, ?, ?, ?, ?)
        """, (
            username,
            email,
            password_hash,
            password_hash,
            salt
        ))

        conn.commit()

        return True, "Registration successful!"

    except sqlite3.IntegrityError:

        return False, "Email already registered."

    except sqlite3.Error as error:

        return False, f"Database error: {error}"

    finally:

        conn.close()


# =====================================================
# 7. LOGIN
# =====================================================

def login_user(email, password):

    email = email.strip().lower()

    conn = sqlite3.connect(DB_PATH)

    try:

        user = conn.execute("""
            SELECT username, email, password_hash, salt
            FROM users
            WHERE email = ?
        """, (email,)).fetchone()

        if user is None:
            return False, None

        username, user_email, stored_hash, salt = user

        if not stored_hash or not salt:
            return False, None

        entered_hash = hash_password(password, salt)

        if secrets.compare_digest(entered_hash, stored_hash):

            return True, {
                "username": username,
                "email": user_email
            }

        return False, None

    finally:

        conn.close()


# =====================================================
# 8. FORGOT PASSWORD
# =====================================================

def create_reset_code(email):

    email = email.strip().lower()

    conn = sqlite3.connect(DB_PATH)

    user = conn.execute(
        "SELECT id FROM users WHERE email=?",
        (email,)
    ).fetchone()

    if user is None:

        conn.close()
        return None

    reset_code = str(secrets.randbelow(900000) + 100000)

    salt = secrets.token_hex(16)

    code_hash = hash_password(reset_code, salt)

    stored_hash = salt + ":" + code_hash

    expires_at = time.time() + 300

    conn.execute("""
        INSERT OR REPLACE INTO password_resets
        (email, reset_code_hash, expires_at)
        VALUES (?, ?, ?)
    """, (
        email,
        stored_hash,
        expires_at
    ))

    conn.commit()
    conn.close()

    return reset_code


def reset_password(email, reset_code, new_password):

    email = email.strip().lower()

    if len(new_password) < 6:

        return False, "Password must contain at least 6 characters."

    conn = sqlite3.connect(DB_PATH)

    record = conn.execute("""
        SELECT reset_code_hash, expires_at
        FROM password_resets
        WHERE email=?
    """, (email,)).fetchone()

    if record is None:

        conn.close()
        return False, "Generate a reset code first."

    stored_value, expires_at = record

    if time.time() > expires_at:

        conn.execute(
            "DELETE FROM password_resets WHERE email=?",
            (email,)
        )

        conn.commit()
        conn.close()

        return False, "Reset code expired."

    try:

        salt, stored_hash = stored_value.split(":", 1)

    except ValueError:

        conn.close()
        return False, "Invalid reset code record."

    entered_hash = hash_password(reset_code, salt)

    if not secrets.compare_digest(entered_hash, stored_hash):

        conn.close()
        return False, "Incorrect reset code."

    new_salt = secrets.token_hex(16)

    new_hash = hash_password(new_password, new_salt)

    try:

        conn.execute("""
            UPDATE users
            SET password=?, password_hash=?, salt=?
            WHERE email=?
        """, (
            new_hash,
            new_hash,
            new_salt,
            email
        ))

        conn.execute(
            "DELETE FROM password_resets WHERE email=?",
            (email,)
        )

        conn.commit()

        return True, "Password changed successfully!"

    except sqlite3.Error as error:

        return False, f"Database error: {error}"

    finally:

        conn.close()


# =====================================================
# 9. SESSION STATE
# =====================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "email" not in st.session_state:
    st.session_state.email = ""

if "reset_email" not in st.session_state:
    st.session_state.reset_email = ""

if "demo_reset_code" not in st.session_state:
    st.session_state.demo_reset_code = None


# =====================================================
# 10. AUTHENTICATION PAGE
# =====================================================

def authentication_page():

    st.markdown(
        "<h1 style='text-align:center;'>🚀 AI POWERED CUSTOMER CHURN PREDICTION</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;'>Customer Churn Prediction & Retention Intelligence System</p>",
        unsafe_allow_html=True
    )

    st.write("")

    left, center, right = st.columns([1, 1.5, 1])

    with center:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        page = st.selectbox(
    "Choose an option",
    ["Login", "Register", "Forgot Password"],
    index=0
)

        if page == "Login":

            st.subheader("Welcome Back 👋")

            with st.form("login_form"):

                email = st.text_input("Email Address")

                password = st.text_input(
                    "Password",
                    type="password"
                )

                submitted = st.form_submit_button(
                    "Login",
                    use_container_width=True
                )

                if submitted:

                    success, user = login_user(email, password)

                    if success:

                        st.session_state.logged_in = True
                        st.session_state.username = user["username"]
                        st.session_state.email = user["email"]

                        st.success("Login successful!")

                        st.rerun()

                    else:

                        st.error("Invalid email or password.")

        elif page == "Register":

            st.subheader("Create Your Account")

            with st.form("register_form"):

                username = st.text_input("Full Name")

                email = st.text_input("Email Address")

                password = st.text_input(
                    "Password",
                    type="password"
                )

                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password"
                )

                submitted = st.form_submit_button(
                    "Register",
                    use_container_width=True
                )

                if submitted:

                    if password != confirm_password:

                        st.error("Passwords do not match.")

                    else:

                        success, message = register_user(
                            username,
                            email,
                            password
                        )

                        if success:

                            st.success(message)
                            st.info("Please select Login to continue.")

                        else:

                            st.error(message)

        else:

            st.subheader("Reset Your Password 🔐")

            email = st.text_input(
                "Registered Email",
                value=st.session_state.reset_email
            )

            if st.button("Generate Reset Code"):

                if not email.strip():

                    st.error("Enter your email address.")

                else:

                    code = create_reset_code(email)

                    if code:

                        st.session_state.reset_email = email.strip().lower()
                        st.session_state.demo_reset_code = code

                        st.success("Reset code generated.")

                    else:

                        st.error("Email not registered.")

            if st.session_state.demo_reset_code:

                st.info(
                    "Demo Reset Code: "
                    + st.session_state.demo_reset_code
                )

                st.caption(
                    "For demonstration only. This code is not emailed."
                )

            with st.form("reset_form"):

                reset_code = st.text_input("Enter Reset Code")

                new_password = st.text_input(
                    "New Password",
                    type="password"
                )

                confirm_password = st.text_input(
                    "Confirm New Password",
                    type="password"
                )

                submitted = st.form_submit_button(
                    "Reset Password",
                    use_container_width=True
                )

                if submitted:

                    if new_password != confirm_password:

                        st.error("Passwords do not match.")

                    elif not st.session_state.reset_email:

                        st.error("Generate a reset code first.")

                    else:

                        success, message = reset_password(
                            st.session_state.reset_email,
                            reset_code,
                            new_password
                        )

                        if success:

                            st.success(message)

                            st.session_state.demo_reset_code = None
                            st.session_state.reset_email = ""

                        else:

                            st.error(message)

        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# 11. MODEL TRAINING
# =====================================================

@st.cache_resource
def train_model():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            "Dataset not found. Keep the Telco CSV in the project folder."
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    X = df[required_columns[:-1]].copy()

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    numeric_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return (
        df,
        model,
        X_test,
        y_test,
        predictions,
        accuracy,
        required_columns[:-1]
    )


# =====================================================
# 12. SIDEBAR NAVIGATION
# =====================================================

def sidebar_navigation():

    st.sidebar.title("🚀 CHURN AI")

    st.sidebar.write(
        f"Welcome, **{st.session_state.username}**"
    )

    st.sidebar.caption(st.session_state.email)

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Customer Prediction",
            "Batch Prediction",
            "Churn Analysis"
        ]
    )

    st.sidebar.markdown("---")

    if st.sidebar.button("Logout", use_container_width=True):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.email = ""

        st.rerun()

    return page


# =====================================================
# 13. HOME DASHBOARD
# =====================================================

def home_page(df, accuracy):

    st.title("📊 Dashboard")

    st.write(
        "Welcome to the AI Powered Customer Churn "
        "Prediction & Retention Intelligence System."
    )

    total = len(df)

    churned = (df["Churn"] == "Yes").sum()

    retained = (df["Churn"] == "No").sum()

    churn_rate = churned / total * 100 if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total:,}")

    col2.metric("Churned Customers", f"{churned:,}")

    col3.metric("Retained Customers", f"{retained:,}")

    col4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        churn_data = pd.DataFrame({
            "Customer Status": ["Churned", "Retained"],
            "Count": [churned, retained]
        })

        fig = px.pie(
            churn_data,
            names="Customer Status",
            values="Count",
            hole=0.55,
            title="Customer Churn Distribution"
        )

        fig.update_layout(
            paper_bgcolor="#080d1c",
            plot_bgcolor="#080d1c",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Model Information")

        st.write("Algorithm: Random Forest Classifier")

        st.write("Training Data: 80%")

        st.write("Testing Data: 20%")

        st.write("Number of Estimators: 100")

        st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)


# =====================================================
# 14. INDIVIDUAL CUSTOMER PREDICTION
# =====================================================

def prediction_page(model, feature_columns):

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Enter customer details to predict whether "
        "the customer may leave the company."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox("Gender", ["Male", "Female"])

            senior = st.selectbox("Senior Citizen", [0, 1])

            partner = st.selectbox("Partner", ["Yes", "No"])

            dependents = st.selectbox("Dependents", ["Yes", "No"])

            tenure = st.number_input(
                "Tenure (Months)", 0, 100, 12
            )

            phone = st.selectbox("Phone Service", ["Yes", "No"])

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges", min_value=0.0, value=70.0
            )

            total = st.number_input(
                "Total Charges", min_value=0.0, value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        customer = customer[feature_columns]

        prediction = model.predict(customer)[0]

        probability = model.predict_proba(customer)[0][1]

        st.markdown("---")

        st.subheader("Prediction Result")

        left, right = st.columns(2)

        with left:

            if prediction == 1:

                st.error("⚠️ Customer is likely to CHURN.")

            else:

                st.success("✅ Customer is likely to STAY.")

        with right:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(float(probability))

        if prediction == 1:

            st.warning(
                "Retention Suggestion: Offer a suitable discount, "
                "improve customer support and contact the customer."
            )

        else:

            st.info(
                "Retention Suggestion: Continue providing good "
                "service and maintain customer satisfaction."
            )


# =====================================================
# 15. BATCH PREDICTION
# =====================================================

def batch_prediction_page(model, feature_columns):

    st.title("📁 Batch Customer Prediction")

    st.write(
        "Upload a CSV file to predict churn for multiple customers."
    )

    st.info(
        "The CSV must contain all 19 feature columns. "
        "customerID and Churn are optional."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Dataset")

            st.dataframe(
                uploaded_df.head(),
                use_container_width=True
            )

            missing = [
                col for col in feature_columns
                if col not in uploaded_df.columns
            ]

            if missing:

                st.error(
                    "Missing columns: " + ", ".join(missing)
                )

                return

            input_df = uploaded_df[feature_columns].copy()

            input_df["TotalCharges"] = pd.to_numeric(
                input_df["TotalCharges"],
                errors="coerce"
            )

            predictions = model.predict(input_df)

            probabilities = model.predict_proba(input_df)[:, 1]

            result_df = uploaded_df.copy()

            result_df["Churn Prediction"] = [
                "Yes" if value == 1 else "No"
                for value in predictions
            ]

            result_df["Churn Probability (%)"] = (
                probabilities * 100
            ).round(2)

            st.markdown("---")

            st.subheader("Prediction Results")

            total = len(result_df)

            churned = int((predictions == 1).sum())

            retained = total - churned

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Customers", total)

            col2.metric("Predicted Churn", churned)

            col3.metric("Predicted Retained", retained)

            st.dataframe(
                result_df,
                use_container_width=True
            )

            csv_data = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Prediction Results",
                data=csv_data,
                file_name="customer_churn_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as error:

            st.error(f"Unable to process CSV: {error}")


# =====================================================
# 16. CHURN ANALYSIS
# =====================================================

def churn_analysis_page(df, y_test, predictions, accuracy):

    st.title("📈 Churn Analysis")

    st.write(
        "Explore customer churn patterns and model performance."
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("Churn by Contract")

        contract_data = (
            df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Contract Type vs Churn"
        )

        fig.update_layout(
            paper_bgcolor="#080d1c",
            plot_bgcolor="#080d1c",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("Churn by Internet Service")

        internet_data = (
            df.groupby(["InternetService", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            internet_data,
            x="InternetService",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Internet Service vs Churn"
        )

        fig.update_layout(
            paper_bgcolor="#080d1c",
            plot_bgcolor="#080d1c",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("Churn by Tenure")

        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=20,
            title="Customer Tenure Distribution"
        )

        fig.update_layout(
            paper_bgcolor="#080d1c",
            plot_bgcolor="#080d1c",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("Churn by Payment Method")

        payment_data = (
            df.groupby(["PaymentMethod", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            payment_data,
            x="PaymentMethod",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Payment Method vs Churn"
        )

        fig.update_layout(
            paper_bgcolor="#080d1c",
            plot_bgcolor="#080d1c",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        title="Model Confusion Matrix",
        aspect="auto"
    )

    fig.update_layout(
        paper_bgcolor="#080d1c",
        plot_bgcolor="#080d1c",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )


# =====================================================
# 17. MAIN APPLICATION
# =====================================================

def main():

    if not st.session_state.logged_in:

        authentication_page()

        return

    selected_page = sidebar_navigation()

    try:

        (
            df,
            model,
            X_test,
            y_test,
            predictions,
            accuracy,
            feature_columns
        ) = train_model()

    except Exception as error:

        st.error(str(error))

        st.info(
            "Check that your Telco CSV dataset is in the "
            "same folder as app.py."
        )

        return

    if selected_page == "Home":

        home_page(df, accuracy)

    elif selected_page == "Customer Prediction":

        prediction_page(model, feature_columns)

    elif selected_page == "Batch Prediction":

        batch_prediction_page(model, feature_columns)

    elif selected_page == "Churn Analysis":

        churn_analysis_page(
            df,
            y_test,
            predictions,
            accuracy
        )


# =====================================================
# 18. START APPLICATION
# =====================================================

if __name__ == "__main__":
    main()
