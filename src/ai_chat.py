"""
ai_chat.py
Prediction-context AI chat, powered by Google's Gemini API.
Reads the API key from an environment variable / Streamlit secrets - never
hardcode it in source. Set GEMINI_API_KEY in your .env file or in
.streamlit/secrets.toml (as GEMINI_API_KEY = "...").

Required env var (set in .env or Streamlit Secrets):
  GEMINI_API_KEY
"""

import os
import time

from google import genai
from google.genai import types, errors

DISCLAIMER = "This is general information, please consult a doctor."

MODEL_NAME = "gemini-3.5-flash"

# Gemini is occasionally overloaded and returns a 503 UNAVAILABLE - this is
# transient (the service says so itself: "usually temporary"), so it's
# worth a couple of quick retries before giving up and showing the user a
# friendly message instead of a raw traceback.
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2  # 2s, 4s, 8s backoff between attempts

BUSY_FALLBACK_MESSAGE = (
    "The AI assistant is receiving unusually high demand right now and "
    "couldn't respond after a few attempts. Please try asking again in a "
    "minute or two."
)

SYSTEM_PROMPT_TEMPLATE = """You are a helpful medical information assistant inside the MediPredict app.

Context for this conversation:
- Disease screened: {disease}
- Model prediction: {result_text} (confidence: {confidence}%)
- Patient's input values: {input_data}

Rules you must follow:
- Only give general health information and explain what the prediction/features mean.
- Never give a definitive diagnosis - only a licensed doctor can diagnose.
- Keep answers concise and easy to understand for a non-medical person.
- Every response must end with this exact disclaimer on its own line: "{disclaimer}"
"""


def _read_env_file_key(key_name):
    """Reads a KEY=value line directly out of <project root>/.env, bypassing
    python-dotenv and os.environ entirely. This file lives in src/, so the
    project root is one directory up. Doing this ourselves, freshly, every
    time is the most bulletproof option - it doesn't depend on
    load_dotenv()'s environment-variable side effect surviving across
    however Streamlit's runtime handles reruns/sessions, which is what kept
    breaking here."""
    project_root_env = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".env")
    )
    if not os.path.isfile(project_root_env):
        return None
    try:
        with open(project_root_env, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key_name:
                    return v.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def get_client():
    # NOTE: os.getenv() and st.secrets.get() both take the *name* of the
    # variable/secret to look up (e.g. "GEMINI_API_KEY") - a real key value
    # must never appear in source code, since anything committed or shared
    # from here is effectively public. If you had a real key pasted in
    # either of these two lookups before, treat that key as compromised and
    # rotate/revoke it in Google AI Studio / Cloud Console now, then put the
    # new one only in your .env file or .streamlit/secrets.toml.
    api_key = os.getenv("GEMINI_API_KEY") or _read_env_file_key("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to a .env file in your project "
            "root (GEMINI_API_KEY=...) for local runs, or to "
            ".streamlit/secrets.toml (GEMINI_API_KEY = \"...\") for Streamlit "
            "Cloud."
        )
    return genai.Client(api_key=api_key)


def _to_gemini_contents(chat_history):
    """Converts {"role": "user"/"assistant", "content": str} history (the
    format the rest of the app already uses) into Gemini's expected
    Content objects - Gemini calls the assistant's role "model", not
    "assistant"."""
    contents = []
    for turn in chat_history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])])
        )
    return contents


def _generate_with_retry(client, model, contents, config):
    """Calls Gemini, retrying a few times with backoff on transient server
    errors (503 UNAVAILABLE, 500 INTERNAL, 429 RESOURCE_EXHAUSTED). Client
    errors (bad API key, invalid request, etc.) are not transient and are
    raised immediately instead of being retried."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except errors.ServerError as e:
            # 503 UNAVAILABLE / 500 INTERNAL - the service itself is
            # overloaded or having a bad moment. Worth retrying.
            last_error = e
        except errors.APIError as e:
            # Anything with a 429 (rate limit) is also worth a brief retry;
            # other client-side errors (401/400/etc.) are not, so re-raise.
            if getattr(e, "code", None) == 429:
                last_error = e
            else:
                raise

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** attempt))

    raise last_error


def ask_ai(disease, prediction, confidence, input_data, user_question, chat_history=None):
    """
    disease: display name, e.g. "Diabetes"
    prediction: int (0 or 1)
    confidence: float
    input_data: dict of the patient's input values
    user_question: str, the free-text question from the user
    chat_history: optional list of {"role": "user"/"assistant", "content": str} from earlier turns

    Returns: str (the assistant's reply, disclaimer included). If Gemini is
    unavailable after retries, returns a friendly fallback message instead
    of raising, so a busy AI backend doesn't crash the chat UI.
    """
    try:
        client = get_client()

        result_text = "High Risk" if prediction == 1 else "Low Risk"
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            disease=disease,
            result_text=result_text,
            confidence=confidence,
            input_data=input_data,
            disclaimer=DISCLAIMER,
        )

        contents = _to_gemini_contents(chat_history) if chat_history else []
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_question)]))

        response = _generate_with_retry(
            client,
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=1024,
            ),
        )

        reply = response.text or ""

        if DISCLAIMER not in reply:
            reply = f"{reply}\n\n{DISCLAIMER}"

        return reply

    except errors.ServerError:
        return BUSY_FALLBACK_MESSAGE
    except errors.APIError:
        return BUSY_FALLBACK_MESSAGE


if __name__ == "__main__":
    answer = ask_ai(
        disease="Diabetes",
        prediction=1,
        confidence=82.5,
        input_data={"Glucose": 150, "BMI": 29.4, "Age": 45},
        user_question="What does my glucose level mean?",
    )
    print(answer)