"""
email_alert.py
Sends an email alert when a prediction indicates high risk.
Reads credentials from environment variables / Streamlit secrets - never hardcode.

Required env vars (set in .env or Streamlit Secrets):
  EMAIL_SENDER      - the Gmail address MediPredict sends from
  EMAIL_APP_PASSWORD - a Gmail App Password (not your normal password)
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

DISCLAIMER = "This is general information, please consult a doctor."


def send_alert_email(to_email, user_name, disease, confidence, tips):
    """
    to_email: recipient's email address
    user_name: str
    disease: display name, e.g. "Heart Disease"
    confidence: float (0-100)
    tips: list[str]
    Returns: (success: bool, message: str)
    """
    sender = os.getenv("EMAIL_SENDER")
    app_password = os.getenv("EMAIL_APP_PASSWORD")

    if not sender or not app_password:
        return False, "Email credentials not configured (EMAIL_SENDER / EMAIL_APP_PASSWORD)."

    subject = f"MediPredict Alert: High Risk Detected for {disease}"

    tips_html = "".join(f"<li>{tip}</li>" for tip in tips)
    body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #0E2B2A;">
        <h2 style="color: #D9603F;">High Risk Alert - {disease}</h2>
        <p>Hi {user_name},</p>
        <p>Your recent MediPredict screening for <b>{disease}</b> indicates a
        <b>high risk</b> result with <b>{confidence}%</b> confidence.</p>
        <h3>Suggestions</h3>
        <ul>{tips_html}</ul>
        <p style="font-size: 12px; color: #4B6461;"><i>{DISCLAIMER}</i></p>
      </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, to_email, msg.as_string())
        return True, "Alert email sent successfully."
    except smtplib.SMTPAuthenticationError:
        return False, "Email authentication failed - check EMAIL_SENDER / EMAIL_APP_PASSWORD."
    except Exception as e:
        return False, f"Failed to send email: {e}"


if __name__ == "__main__":
    ok, message = send_alert_email(
        to_email="test@example.com",
        user_name="Test User",
        disease="Heart Disease",
        confidence=88.0,
        tips=["Reduce salt intake.", "Exercise regularly."],
    )
    print(ok, message)
