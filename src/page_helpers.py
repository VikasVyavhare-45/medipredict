"""
page_helpers.py
Shared workflow that every disease page (pages/1_Diabetes.py, pages/2_Heart.py, ...)
calls after collecting form inputs. Keeps app.py / each page thin - all the
"what happens after the user hits Predict" logic lives here, once.

The result is rendered as a small Power-BI-style dashboard: KPI tiles for
risk/confidence, a radial confidence gauge, a horizontal bar chart for
feature impact, suggestions, actions, and a real multi-turn AI chat thread
below. All of it is plain HTML+CSS (no plotly/matplotlib dependency) so it
renders anywhere Streamlit does.
"""

import os
import streamlit as st
import joblib

from predict import predict_disease, MODELS_DIR, BEST_MODEL
from suggestions import get_suggestions
from explainability import explain_prediction
from pdf_report import generate_pdf_report
from email_alert import send_alert_email
from ai_chat import ask_ai
from database import save_prediction


def get_feature_order(disease, model_name=None):
    """Returns the exact ordered list of columns the disease's scaler expects."""
    model_name = model_name or BEST_MODEL[disease]
    scaler = joblib.load(f"{MODELS_DIR}/{disease}/scaler.pkl")
    return list(scaler.feature_names_in_)


def prepare_input_dict(disease, direct_values, dummy_flags=None):
    """
    Builds a complete, correctly-shaped input_dict for predict_disease() /
    explain_prediction(), without needing to hardcode each disease's full
    (possibly one-hot-encoded) column list.

    direct_values: {column_name: value} for plain numeric/binary fields that
                   exist as their own column in scaler.feature_names_in_
                   (e.g. {"age": 55, "sex": 1, "chol": 230})
    dummy_flags:   list of one-hot column names to set to 1, matching how
                   pd.get_dummies(..., drop_first=True) named them at training
                   time (e.g. ["cp_atypical angina", "thal_fixed defect"]).
                   Any column not in direct_values or dummy_flags defaults to 0.

    Any dummy_flags entry that doesn't match an actual training column is
    silently ignored (it means that value was the drop_first baseline, which
    is correct - the baseline category is represented by all-zeros).
    """
    order = get_feature_order(disease)
    input_dict = {col: 0 for col in order}

    for col, val in direct_values.items():
        if col in input_dict:
            input_dict[col] = val

    if dummy_flags:
        for flag in dummy_flags:
            if flag in input_dict:
                input_dict[flag] = 1

    return input_dict


# ---------------------------------------------------------------------------
# Dashboard styling (pure HTML/CSS, matching the MediPredict forest theme)
# ---------------------------------------------------------------------------

def _dashboard_css():
    st.markdown(
        """
        <style>
        .mp-card {
            background:#FFFFFF; border:1px solid #DCEBDF; border-radius:14px;
            box-shadow:0 2px 8px rgba(6,38,27,0.05);
            transition: transform .15s ease, box-shadow .2s ease;
            position:relative; overflow:hidden;
        }
        .mp-card::before {
            content:""; position:absolute; top:0; left:0; right:0; height:3px;
            background:linear-gradient(90deg, var(--mp-accent,#1F8A56), transparent);
        }
        .mp-card:hover { transform:translateY(-2px); box-shadow:0 12px 24px -10px rgba(6,38,27,0.18); }
        .mp-kpi { padding:18px 20px 18px 20px; height:100%; }
        .mp-kpi-top { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
        .mp-kpi-icon {
            width:30px; height:30px; border-radius:9px; display:flex; align-items:center;
            justify-content:center; font-size:15px; background:color-mix(in srgb, var(--mp-accent,#1F8A56) 14%, white);
        }
        .mp-kpi .mp-label { font-size:11px; text-transform:uppercase; letter-spacing:.07em;
            color:#6B8A80; font-weight:700; }
        .mp-kpi .mp-value { font-family:'IBM Plex Mono',monospace; font-size:24px; font-weight:700; line-height:1.15; }
        .mp-kpi .mp-sub { font-size:12px; color:#8AA69D; margin-top:6px; }
        .mp-section-tag { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
            color:#6B8A80; font-weight:700; margin-bottom:6px; }

        /* Risk meter */
        .mp-meter-track {
            position:relative; height:14px; border-radius:8px; margin:14px 2px 8px 2px;
            background:linear-gradient(90deg, #1F8A56 0%, #E8B23D 55%, #D9603F 100%);
        }
        .mp-meter-dot {
            position:absolute; top:-5px; width:24px; height:24px; border-radius:50%;
            background:#FFFFFF; border:3px solid #072B1E; transform:translateX(-50%);
            box-shadow:0 3px 8px rgba(6,38,27,0.35);
        }
        .mp-meter-labels { display:flex; justify-content:space-between; font-size:10.5px;
            color:#8AA69D; text-transform:uppercase; letter-spacing:.05em; padding:0 2px; }

        /* Chat thread - bubble background is intentionally always this
           light green (not theme-adaptive, same as the rest of the
           dashboard's white mp-cards), so the text inside must be pinned
           to a dark ink color too. Left unset, it was inheriting
           Streamlit's own light-colored dark-mode text, which is nearly
           invisible against this light bubble. */
        div[data-testid="stChatMessage"] {
            background:#F4FAF5 !important; border:1px solid #DCEBDF !important;
            border-radius:12px !important; padding:4px 6px !important; margin-bottom:6px;
        }
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] div,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessageContent"],
        div[data-testid="stChatMessageContent"] * {
            color:#072B1E !important;
        }
        div[data-testid="stChatInput"] textarea { border-radius:10px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label, value, sublabel, color, icon=""):
    st.markdown(
        f"""
        <div class="mp-card mp-kpi" style="--mp-accent:{color};">
          <div class="mp-kpi-top">
            <div class="mp-kpi-icon">{icon}</div>
            <div class="mp-label">{label}</div>
          </div>
          <div class="mp-value" style="color:{color};">{value}</div>
          <div class="mp-sub">{sublabel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _risk_meter(confidence, color):
    pos = max(2, min(98, confidence))
    st.markdown(
        f"""
        <div class="mp-card" style="padding:20px 24px;">
          <div class="mp-section-tag">📏 Risk meter</div>
          <div class="mp-meter-track">
            <div class="mp-meter-dot" style="left:{pos}%; border-color:{color};"></div>
          </div>
          <div class="mp-meter-labels"><span>Low</span><span>Moderate</span><span>High</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _confidence_gauge(confidence, color):
    deg = max(0, min(100, confidence)) * 3.6
    st.markdown(
        f"""
        <div class="mp-card" style="padding:22px;height:100%;display:flex;flex-direction:column;
        align-items:center;justify-content:center;">
          <div style="width:148px;height:148px;border-radius:50%;
          background:conic-gradient({color} {deg}deg, #E8EEEA {deg}deg 360deg);
          display:flex;align-items:center;justify-content:center;">
            <div style="width:110px;height:110px;border-radius:50%;background:#FFFFFF;
            display:flex;flex-direction:column;align-items:center;justify-content:center;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:700;
              color:{color};">{confidence}%</div>
              <div style="font-size:10px;color:#8AA69D;letter-spacing:.04em;">CONFIDENCE</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _feature_impact_panel(explanation_df):
    max_abs = float(explanation_df["abs_impact"].max()) or 1.0
    rows_html = ""
    for _, row in explanation_df.iterrows():
        pct = max(4, (float(row["abs_impact"]) / max_abs) * 100)
        pushes_up = float(row["shap_value"]) > 0
        color = "#D9603F" if pushes_up else "#1F8A56"
        rows_html += f"""
        <div style="margin-bottom:13px;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;
          font-size:12.5px;margin-bottom:5px;">
            <span style="font-weight:600;color:#072B1E;">{row['feature']}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:{color};">
              {float(row['shap_value']):+.4f}</span>
          </div>
          <div style="background:#EDF3EF;border-radius:6px;height:8px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:6px;"></div>
          </div>
        </div>
        """
    st.markdown(
        f"""
        <div class="mp-card" style="padding:20px 22px;height:100%;">
          <div class="mp-section-tag">📊 Top contributing factors</div>
          {rows_html}
          <div style="display:flex;gap:16px;font-size:11px;color:#8AA69D;margin-top:4px;">
            <span><span style="display:inline-block;width:9px;height:9px;background:#D9603F;
            border-radius:2px;margin-right:5px;"></span>pushes risk up</span>
            <span><span style="display:inline-block;width:9px;height:9px;background:#1F8A56;
            border-radius:2px;margin-right:5px;"></span>pushes risk down</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _suggestions_card(tips, disclaimer, color):
    tips_html = "".join(
        f"<li style='margin-bottom:8px; padding-left:4px;'>{tip}</li>" for tip in tips
    )
    st.markdown(
        f"""
        <div class="mp-card" style="padding:20px 24px;--mp-accent:{color};">
          <div class="mp-section-tag">💡 Suggestions</div>
          <ul style="margin:0 0 10px 0;padding:0 0 0 4px;list-style:none;color:#233F35;
          font-size:13.5px;line-height:1.6;">
            {tips_html}
          </ul>
          <div style="font-size:11.5px;color:#8AA69D;">{disclaimer}</div>
        </div>
        <style>
        .mp-card ul li::before {{
            content:"●"; color:{color}; font-size:8px; margin-right:10px; vertical-align:middle;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------

def run_prediction_workflow(disease_key, disease_display_name, input_dict,
                             background_data=None, user_email=None):
    """
    Call this from a disease page after the user submits their form.

    disease_key: e.g. "diabetes" - must match models/<disease_key>/ folder name
    disease_display_name: e.g. "Diabetes" - shown to the user / in PDF & email
    input_dict: {feature_name: value, ...} collected from the Streamlit form
    background_data: only needed for the one disease whose best model isn't
                      tree-based (breast_cancer -> svm) - pass a sample of
                      scaled training rows the first time
    user_email: if set and risk is high, offers to send an alert email there
    """
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to get a prediction.")
        return

    user = st.session_state.user
    _dashboard_css()

    result = predict_disease(disease_key, input_dict)
    suggestions = get_suggestions(disease_key, result["prediction"])

    is_high_risk = result["prediction"] == 1
    risk_color = "#D9603F" if is_high_risk else "#1F8A56"
    risk_text = "High Risk" if is_high_risk else "Low Risk"
    risk_icon = "🔴" if is_high_risk else "🟢"

    # Only write to history once per submission, not on every rerun triggered
    # by the PDF/email/Ask buttons below (form_builder resets this flag to
    # False each time the user clicks Predict again).
    saved_flag_key = f"{disease_key}_history_saved"
    if not st.session_state.get(saved_flag_key):
        save_prediction(user["id"], disease_key, input_dict, result["prediction"], result["confidence"])
        st.session_state[saved_flag_key] = True
        # Fresh prediction -> start a fresh chat thread for this result
        st.session_state[f"{disease_key}_chat_history"] = []

    st.markdown("<div class='mp-section-tag'>Result dashboard</div>", unsafe_allow_html=True)
    st.markdown(f"## {disease_display_name} risk overview")

    # ---- KPI row ----
    model_name = BEST_MODEL.get(disease_key, "model")
    model_label = str(model_name).replace("_", " ").title()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Risk Level", f"{risk_icon} {risk_text}", "Model prediction", risk_color, icon="🎯")
    with k2:
        _kpi_card("Confidence", f"{result['confidence']}%", "Prediction certainty", risk_color, icon="📈")
    with k3:
        _kpi_card("Inputs Reviewed", str(len(input_dict)), "Fields used for this screening", "#1F8A56", icon="🧾")
    with k4:
        _kpi_card("Model Used", model_label, "Best-performing algorithm", "#2E6F8E", icon="⚙️")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ---- Risk meter ----
    _risk_meter(result["confidence"], risk_color)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ---- Gauge + feature impact row ----
    g1, g2 = st.columns([1, 2])
    with g1:
        _confidence_gauge(result["confidence"], risk_color)
    with g2:
        try:
            explanation = explain_prediction(disease_key, input_dict, background_data=background_data, top_n=5)
            _feature_impact_panel(explanation)
        except Exception as e:
            st.caption(f"Explanation unavailable: {e}")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ---- Suggestions ----
    _suggestions_card(suggestions["tips"], suggestions["disclaimer"], risk_color)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ---- Actions ----
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Generate PDF report", key=f"{disease_key}_pdf"):
            path = generate_pdf_report(
                user_name=user["username"],
                disease=disease_display_name,
                prediction=result["prediction"],
                confidence=result["confidence"],
                tips=suggestions["tips"],
                input_data=input_dict,
            )
            with open(path, "rb") as f:
                st.download_button("Download report", f, file_name=os.path.basename(path))

    with col2:
        if result["prediction"] == 1 and user_email:
            if st.button("📧 Send high-risk alert email", key=f"{disease_key}_email"):
                ok, msg = send_alert_email(
                    to_email=user_email,
                    user_name=user["username"],
                    disease=disease_display_name,
                    confidence=result["confidence"],
                    tips=suggestions["tips"],
                )
                st.success(msg) if ok else st.error(msg)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ---- Ask AI — a real multi-turn chat thread, not a single Q&A box ----
    chat_key = f"{disease_key}_chat_history"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    st.markdown("<div class='mp-section-tag'>💬 Ask the AI about your result</div>", unsafe_allow_html=True)

    with st.container(border=True):
        if not st.session_state[chat_key]:
            st.caption("Ask anything about this result — e.g. \"why is my glucose flagged?\" or "
                       "\"what should I do next?\". You can ask more than one question.")
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🩺"):
                st.write(msg["content"])

    question = st.chat_input("Type a question and press Enter…", key=f"{disease_key}_chat_input")
    if question:
        st.session_state[chat_key].append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            answer = ask_ai(
                disease=disease_display_name,
                prediction=result["prediction"],
                confidence=result["confidence"],
                input_data=input_dict,
                user_question=question,
            )
        st.session_state[chat_key].append({"role": "assistant", "content": answer})
        st.rerun()

    return result
