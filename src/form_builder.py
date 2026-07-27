"""
form_builder.py
Builds a professional-looking Streamlit input form for ANY of the 10
diseases, driven directly by that disease's saved scaler
(`scaler.feature_names_in_`). This is the key design choice: instead of
hand-typing form fields and hoping they match what the model was trained
on (risky - one typo silently breaks predictions), the form is generated
FROM the same scaler predict.py and explainability.py already trust.

Add friendlier labels/icons per disease below (DISEASE_META) purely for
display - the actual input keys always come from the scaler, so
correctness never depends on keeping that metadata in sync.
"""

import os
import joblib
import streamlit as st

from page_helpers import run_prediction_workflow

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

DISEASE_META = {
    "diabetes": {"title": "Diabetes Screening", "icon": "🩸",
                 "desc": "Assess diabetes risk from glucose, BMI, and related vitals."},
    "heart": {"title": "Heart Disease Screening", "icon": "❤️",
              "desc": "Assess cardiovascular risk from clinical and ECG indicators."},
    "parkinsons": {"title": "Parkinson's Screening", "icon": "🧠",
                   "desc": "Assess Parkinson's risk from voice measurement biomarkers."},
    "liver": {"title": "Liver Disease Screening", "icon": "🫀",
              "desc": "Assess liver disease risk from liver function panel values."},
    "kidney": {"title": "Kidney Disease Screening", "icon": "💧",
               "desc": "Assess chronic kidney disease risk from renal panel values."},
    "breast_cancer": {"title": "Breast Cancer Screening", "icon": "🎗️",
                       "desc": "Assess risk from tumor cell nuclei measurements."},
    "stroke": {"title": "Stroke Risk Screening", "icon": "🧬",
               "desc": "Assess stroke risk from lifestyle and health indicators."},
    "hepatitis": {"title": "Hepatitis Screening", "icon": "🩺",
                  "desc": "Assess hepatitis risk from clinical and lab indicators."},
    "thyroid": {"title": "Thyroid Disorder Screening", "icon": "🦋",
                "desc": "Assess thyroid disorder risk from hormone panel values."},
    "lung_cancer": {"title": "Lung Cancer Screening", "icon": "🫁",
                     "desc": "Assess lung cancer risk from symptoms and history."},
}

# Fields that are naturally yes/no or male/female in these datasets - shown
# as a select box instead of a raw number box, but still stored as 0/1
# (matching how they were encoded during training).
BINARY_FIELD_HINTS = {
    "sex": ("Female", "Male"),
    "Sex": ("Female", "Male"),
    "gender": ("Female", "Male"),
    "Gender": ("Female", "Male"),
    "GENDER": ("Female", "Male"),
    "htn": ("No", "Yes"), "dm": ("No", "Yes"), "cad": ("No", "Yes"),
    "pe": ("No", "Yes"), "ane": ("No", "Yes"), "appet": ("Poor", "Good"),
    "hypertension": ("No", "Yes"), "heart_disease": ("No", "Yes"),
    "smoking": ("No", "Yes"), "SMOKING": ("No", "Yes"),
    "fbs": ("No", "Yes"), "exang": ("No", "Yes"),
}

# Fields that are naturally whole numbers (counts, ages, heart rates, vessel
# counts, etc). Without this, every field - including Age and Pregnancies -
# was shown with the raw training-data MEAN as its default (e.g. Age
# "33.37", Pregnancies "3.82") and a 2-decimal-place input box, since
# _field_widget() had no way to tell an integer field apart from a truly
# continuous one like BMI or DiabetesPedigreeFunction.
#
# NOTE: only the "diabetes" block below has been verified against an
# actual scaler.feature_names_in_ (from your notebook). The other 9 are
# filled in from the standard public UCI/Kaggle datasets these diseases
# normally use - if your notebook renamed, dropped, or one-hot-encoded
# any of these columns differently, update the matching set below to
# match your real scaler.feature_names_in_ (print it once to check).
INTEGER_FIELD_HINTS = {
    # --- diabetes (Pima Indians) - confirmed against your scaler ---
    "Age", "Pregnancies",

    # --- heart (UCI Heart Disease) ---
    "age", "trestbps", "chol", "thalach", "ca", "cp", "restecg", "slope", "thal",

    # --- liver (Indian Liver Patient Dataset) ---
    "Age", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase",

    # --- kidney (UCI Chronic Kidney Disease) ---
    "age", "bp", "al", "su", "bgr", "bu", "pcv", "wc",

    # --- stroke (Kaggle Stroke Prediction) ---
    "age",

    # --- hepatitis (UCI Hepatitis) ---
    "AGE", "ALK PHOSPHATE", "SGOT", "PROTIME",

    # --- thyroid (UCI Thyroid Disease) ---
    "age",

    # --- lung_cancer (Kaggle Lung Cancer survey dataset) ---
    "AGE",

    # parkinsons and breast_cancer datasets are entirely continuous
    # biomarker/measurement values (voice jitter/shimmer, cell nuclei
    # radius/texture/etc) - no naturally-integer fields, nothing to add.
}


def _apply_page_chrome():
    """Every disease page is its own standalone Streamlit script (that's how
    the pages/ folder works), so app.py's apply_theme() never runs for them -
    each page needs its own copy of the same chrome-hiding + brand CSS, or it
    falls back to Streamlit's bare-default look (visible Deploy button, the
    auto-generated sidebar page list, unstyled buttons/forms).
    Called once at the top of render_disease_form so all 10 pages get it for
    free from this single shared file.

    Colors here are kept in sync by hand with the --root CSS vars defined in
    app.py (search that file for ':root' if you re-theme again):
      --ink #1E1B3D | --ink-soft #5D5680 | --forest-deep #1B1740
      --forest #3D2E8C | --forest-mid #5B3FC4 | --green #7C3AED
      --sage #F2EFFC | --line #E1DCF5
    """
    st.markdown(
        """
        <style>
        /* Same theme-variable palette as app.py's :root block, duplicated
           here because each disease page is its own standalone Streamlit
           script and never runs app.py's apply_theme(). Keep these two
           blocks in sync by hand if the palette changes. */
        :root{
            color-scheme: light dark;
            --ink: #1E1B3D;
            --ink-soft: #5D5680;
            --ink-faint: #8983AD;
            --forest-deep: #1B1740;
            --green: #7C3AED;
            --forest-mid: #5B3FC4;
            --sage: #F2EFFC;
            --card: #FFFFFF;
            --line: #E1DCF5;
            --shadow-ink: rgba(27,23,64,0.4);
            /* --forest-deep is TEXT ink here (title card heading) and
               flips light in dark mode below. --panel-deep is a SEPARATE
               constant used only for the always-dark gradient buttons,
               which carry white text in both themes and must never flip. */
            --panel-deep: #1B1740;
        }
        @media (prefers-color-scheme: dark){
            :root{
                --ink: #EDE9FB;
                --ink-soft: #B3ABD6;
                --ink-faint: #857CAD;
                --forest-deep: #F5F1FF;
                --green: #9D6FF0;
                --forest-mid: #382A78;
                --sage: #100D22;
                --card: #1B1732;
                --line: #322B58;
                --shadow-ink: rgba(0,0,0,0.55);
            }
        }
        .stApp { color-scheme: light dark; }

        /* Hide Streamlit's own chrome */
        #MainMenu { visibility: hidden; }
        .stAppDeployButton { display: none !important; }
        div[data-testid="stToolbarActions"] { display: none !important; }
        footer { visibility: hidden; }
        [data-testid="stSidebarNav"] { display: none !important; }
        header[data-testid="stHeader"] { display: none !important; }

        /* Sidebar removed entirely on these pages - the "Back to Home" link
           and login status now live in the main content instead (see
           _back_to_home_button() at the bottom of render_disease_form). */
        section[data-testid="stSidebar"] { display: none !important; }
        div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
        .block-container { padding-top: 2rem !important; max-width: 100% !important; }

        /* Match the MediPredict purple/violet theme from app.py */
        .stApp { background-color: var(--sage); }
        h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--forest-deep); }
        body, p, div, span, label { font-family: 'Inter', sans-serif; }

        .stButton>button {
            background: linear-gradient(160deg, var(--green), var(--panel-deep));
            color: white; border-radius: 9px; border: none; font-weight: 600;
            padding: 0.5rem 1.2rem; box-shadow: 0 8px 18px -6px var(--shadow-ink);
            transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(160deg, var(--forest-mid), var(--panel-deep));
            box-shadow: 0 14px 26px -6px var(--shadow-ink);
            transform: translateY(-2px);
        }
        .stButton>button:active { transform: translateY(0px); }

        .stAlert { border-radius: 10px; }
        div[data-testid="stForm"] { border-radius: 14px; border: 1px solid var(--line); }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px !important;
            box-shadow: 0 4px 14px rgba(27,23,64,0.05);
        }

        /* Number input boxes - match the card / soft violet border
           used throughout app.py instead of Streamlit's default grey. */
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {
            background: var(--card) !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
        }

        /* The value text actually shown INSIDE a selectbox (e.g. "Male" /
           "Female" once picked) lives in nested spans/divs that the rule
           above doesn't reach - only its outer wrapper does. Left unset,
           that text inherits Streamlit's own internal theme color instead
           of ours. Same for the dropdown's small caret icon. */
        div[data-baseweb="select"] > div * ,
        div[data-baseweb="select"] input {
            color: var(--ink) !important;
            fill: var(--ink-soft) !important;
        }
        div[data-baseweb="select"] svg { fill: var(--ink-soft) !important; }

        /* Open dropdown menu (the list of options after clicking a
           selectbox) renders in its own floating layer, also themed by
           Streamlit's internal light/dark setting rather than our CSS -
           pin it to the same palette so it always stays readable. */
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background: var(--card) !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li,
        ul[data-testid="stSelectboxVirtualDropdown"] li * {
            color: var(--ink) !important;
        }

        /* st.warning() / st.info() / st.error() boxes (e.g. the "please
           log in" notice) - keep text on our own palette, same reasoning
           as above. */
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span,
        div[data-testid="stAlertContentWarning"] p,
        div[data-testid="stAlertContentInfo"] p,
        div[data-testid="stAlertContentError"] p,
        div[data-testid="stAlertContentSuccess"] p {
            color: var(--ink) !important;
        }

        /* Placeholder text inside number/select inputs (the greyed-out
           "hint" value shown before the user types anything) - make it
           visibly muted/italic so it reads clearly as a HINT and not as
           an already-filled real value. Disappears the moment the user
           clicks in and types (native browser/Streamlit placeholder
           behavior - nothing custom needed for that part). */
        div[data-testid="stNumberInput"] input::placeholder {
            color: var(--ink-faint) !important;
            opacity: 0.8 !important;
            font-style: italic;
        }

        /* ---- Keep native Streamlit widgets on our own theme colors ---- */
        /* Same as app.py: Streamlit's own internal theme can otherwise
           render field labels (Pregnancies, Glucose, Age...) and captions
           ("Logged in as ...") in a color that doesn't match our page
           background. This ONLY targets genuinely-native widgets
           (number_input/selectbox labels, captions) - it deliberately does
           NOT touch .stMarkdown / stMarkdownContainer, since the title card
           above is also delivered via st.markdown() and already sets its
           own theme-variable colors. */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .stSelectbox label, .stNumberInput label, .stTextInput label,
        .stCheckbox label, .stRadio label {
            color: var(--ink) !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {
            color: var(--ink-soft) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _back_to_home_button():
    """Rendered at the bottom of every disease page. Replaces the old
    sidebar "Back to Home" link now that the sidebar is hidden entirely -
    this is the only way back to the Home page from a screening page."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    left, _ = st.columns([1, 4])
    with left:
        st.page_link("app.py", label="⬅ Back to Home", icon="🏠", use_container_width=True)


def _load_feature_stats(disease_key):
    scaler = joblib.load(os.path.join(MODELS_DIR, disease_key, "scaler.pkl"))
    if not hasattr(scaler, "feature_names_in_"):
        raise ValueError(
            f"scaler for '{disease_key}' has no feature_names_in_ - "
            f"it must be re-fit on a pandas DataFrame (X_train), not a numpy array."
        )
    return list(scaler.feature_names_in_), list(scaler.mean_)


def _field_widget(feature_name, default_value=0.0):
    """Renders one input widget for a feature, returns its current value.

    The training-set mean is shown only as a PLACEHOLDER (grey hint text),
    not as a real pre-filled value - the box starts empty. The hint
    disappears the instant the user clicks in and starts typing (that's
    native placeholder behavior, same as any "Search..." box). If the user
    never types anything and submits as-is, the mean is still used under
    the hood as the fallback value, so predictions never break on an
    untouched field.

    Requires streamlit >= 1.34 (that's when `placeholder=` was added to
    st.number_input / st.selectbox). If you're on an older version:
        pip install --upgrade streamlit
    """
    if feature_name in BINARY_FIELD_HINTS:
        low_label, high_label = BINARY_FIELD_HINTS[feature_name]
        default_label = high_label if default_value >= 0.5 else low_label
        choice = st.selectbox(
            feature_name.replace("_", " ").title(),
            [low_label, high_label],
            index=None,
            placeholder=default_label,
            key=f"field_{feature_name}",
        )
        if choice is None:
            return 1 if default_value >= 0.5 else 0
        return 1 if choice == high_label else 0

    if feature_name in INTEGER_FIELD_HINTS:
        val = st.number_input(
            feature_name.replace("_", " ").title(),
            value=None,
            step=1,
            format="%d",
            min_value=0,
            placeholder=str(int(round(default_value))),
            key=f"field_{feature_name}",
        )
        return val if val is not None else int(round(default_value))

    val = st.number_input(
        feature_name.replace("_", " ").title(),
        value=None,
        step=0.1,
        format="%.2f",
        placeholder=str(round(float(default_value), 2)),
        key=f"field_{feature_name}",
    )
    return val if val is not None else round(float(default_value), 2)


def render_disease_form(disease_key, user_email_getter=None):
    """
    Call this from a thin pages/<n>_<Disease>.py file - it renders the
    full form + Predict button + result (via run_prediction_workflow).
    """
    _apply_page_chrome()

    meta = DISEASE_META.get(disease_key, {"title": disease_key.title(), "icon": "🩺", "desc": ""})

    st.markdown(
        f"""
        <div style="background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:20px 24px;margin-bottom:20px;">
        <h2 style="margin:0;color:var(--forest-deep);">{meta['icon']} {meta['title']}</h2>
        <p style="color:var(--ink-soft);margin:6px 0 0 0;">{meta['desc']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.get("logged_in"):
        st.warning("Please log in from the Home page to use this screening tool.")
        _back_to_home_button()
        return

    st.caption(f"Logged in as **{st.session_state.user['username']}**")

    feature_names, feature_means = _load_feature_stats(disease_key)

    input_dict = {}
    cols = st.columns(3)
    for i, (feature, mean_val) in enumerate(zip(feature_names, feature_means)):
        with cols[i % 3]:
            input_dict[feature] = _field_widget(feature, default_value=mean_val)

    st.markdown("<br>", unsafe_allow_html=True)

    submitted_key = f"{disease_key}_submitted"
    input_key = f"{disease_key}_saved_input"

    if st.button(f"🔍 Predict {meta['title'].replace(' Screening', '')}", key=f"{disease_key}_predict"):
        st.session_state[submitted_key] = True
        st.session_state[input_key] = input_dict
        st.session_state[f"{disease_key}_history_saved"] = False

    # Re-render the result on every rerun (e.g. after clicking PDF/email/Ask),
    # not just on the exact run where the Predict button was clicked - button
    # click state doesn't persist across Streamlit reruns, but session_state does.
    if st.session_state.get(submitted_key):
        user_email = None
        if user_email_getter:
            user_email = user_email_getter()
        else:
            user_email = st.session_state.user.get("email") if st.session_state.get("user") else None

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            run_prediction_workflow(disease_key, meta["title"].replace(" Screening", ""),
                                     st.session_state[input_key], user_email=user_email)

    _back_to_home_button()
