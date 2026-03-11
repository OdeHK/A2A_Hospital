# 🩺 AI Medical Assistant Agent

## Introduction
AI Medical Assistant Agent is an intelligent system designed to help users identify possible illnesses based on symptoms and schedule appointments with doctors.

The system uses Artificial Intelligence and Natural Language Processing (NLP) to understand user symptoms, provide preliminary health advice, and assist in booking medical consultations.

This project aims to improve healthcare accessibility by allowing users to receive quick medical guidance and appointment scheduling through an AI-powered interface.

---

## 🚀 Features

### 1. Symptom Analysis
Users can describe their symptoms in natural language, and the AI agent will:
- Analyze the symptoms
- Suggest possible diseases
- Provide basic medical advice
- Recommend whether the user should see a doctor

### 2. Medical Advice
The AI agent provides:
- Basic health guidance
- Suggested treatments for minor conditions
- Recommendations to consult a specialist if necessary

⚠️ Disclaimer:  
This system does not replace professional medical diagnosis.

### 3. Doctor Recommendation
Based on the symptoms, the system can:
- Recommend suitable doctors
- Suggest medical specialties
- Show available doctors

### 4. Appointment Booking
Users can:
- View doctor availability
- Choose date and time
- Book appointments

### 5. Chat-based Interaction
Users interact with the AI through a chatbot interface.

---

## 🏗 System Architecture

User  
↓  
Frontend (Web)  
↓  
AI Agent  
├── NLP Processing  
├── Symptom Analysis  
├── Disease Prediction  
↓  
Backend API  
├── User Management  
├── Appointment Management  
├── Doctor Database  
↓  
Database  

---

## 🛠 Technologies Used

Backend
- Python
- Flask
- REST API
- JWT Authentication

AI
- LLM
- Prompt Engineering
- NLP
- Hugging Face
- Langchain

Database
- PostgreSQL / MySQL
- Redis

Frontend
- Html / css

---

## 📂 Project Structure

A2A_Hospital

flask-app/
   blueprints/
   models/
   services/
   static/
   templates/

Agent/
   agents/cost_agent/
   host_agent/
   common/

frontend/
docs/

README.md