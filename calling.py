import time
from twilio.rest import Client
from deepgramvoice import transcribe_audio


# Twilio credentials (Get these from your Twilio Console)
account_sid = "ACxxxxxxxxxxxxxxxxxxxxx"
auth_token = "xxxxxxxxxxxxxxxxxxxxxx"
client = Client(account_sid, auth_token)

# Make an outbound call
call = client.calls.create(
    to="+9191004011xx",
    from_="+19302123xxx",
    twiml="""
        <Response>
            <Say>Hello! This is a test call from Twilio. Please speak after the beep.</Say>
            <Pause length="2"/>
            <Record maxLength="20" playBeep="true" />
        </Response>
    """,
    record=True  # Ensures recording
)



print(f"Call SID: {call.sid}")

time.sleep(50)

recordings = client.recordings.list()
if recordings:
    latest_recording = recordings[0]
    audio_url = f"https://api.twilio.com{latest_recording.uri.replace('.json', '.wav')}"

    print(f"Recording URL: {audio_url}")

    transcription = transcribe_audio(audio_url)
    print("Full Deepgram Response:", transcription)

    # Extract transcript safely
    results = transcription.get("results", {})
    channels = results.get("channels", [])
    if channels:
        alternatives = channels[0].get("alternatives", [])
        transcript = alternatives[0].get("transcript", "No transcript available") if alternatives else "No alternatives found"
    else:
        transcript = "No channels found"

    print("Transcription:", transcript)
else:
    print("No recordings found.")
