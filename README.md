# 🗣️ AI Voice Agent for BFSI - Outbound SDR Automation

## Overview

This project is an AI-driven **Voice Agent** that simulates the role of a **Sales Development Representative (SDR)** for fintech companies. The voice agent autonomously initiates outbound calls in **English and Hindi**, engages with customers using a dynamic conversation flow, captures responses, and facilitates lead generation.

## 🎯 Objective

To build a smart voice-based system that:
- Mimics natural human interaction with clear tone and context awareness.
- Connects to CRM systems to personalize outbound calls.
- Logs calls with transcripts and audio recordings for future analysis.
- Complies with fintech regulatory and data privacy standards.

---

## 🛠️ Key Features

- ✅ Multilingual support (English & Hindi)
- ✅ Natural language understanding and speech synthesis
- ✅ Telephony integration for outbound calling
- ✅ CRM and database connectivity
- ✅ Real-time call logging and transcripts
- ✅ Context-aware conversation with dynamic scripting

---

## 📦 Tech Stack

| Component               | Technology Used                                   |
|------------------------|---------------------------------------------------|
| Programming Language   | Python                                            |
| Speech-to-Text (STT)   | Deepgram / Navana.ai                              |
| Text-to-Speech (TTS)   | ElevenLabs / Smallest.ai                          |
| NLP Engine             | Open-source LLMs (Local or Cloud Hosted)          |
| Telephony Integration  | Twilio / Plivo                                    |
| Backend & APIs         | FastAPI / Flask                                   |
| Database               | PostgreSQL / MongoDB                              |
| Call Recording/Logging | Twilio Call Logs + Audio + Custom Transcriber     |

---

## 🧠 System Architecture

```mermaid
graph TD
    UI[Voice Agent (TTS + STT)]
    Telephony[Telephony API (Twilio/Plivo)]
    CRM[CRM/Database]
    NLP[NLP Engine (LLM)]
    Logger[Logging & Analytics]

    UI --> Telephony
    Telephony --> CRM
    Telephony --> Logger
    Telephony --> NLP
    NLP --> UI
    CRM --> UI
