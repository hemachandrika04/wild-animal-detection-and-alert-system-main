# --- Twilio Configuration ---
# IMPORTANT: Keep this file secure and out of public repositories.
# Fill in your actual Twilio details below.
import os

ACCOUNT_SID = os.getenv("ACCOUNT_SID")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
SENDER_PHONE = os.getenv("SENDER_PHONE")
RECEIVER_PHONE = os.getenv("RECEIVER_PHONE")