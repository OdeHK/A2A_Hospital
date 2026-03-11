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
|--blueprints/
|-- models/
|-- services/
|-- static/
|-- templates/

Agent/
|--agents/
|--host_agent/
|--common/

frontend/
docs/

README.md

## How to run
# Sample Code

This code is used to demonstrate A2A capabilities as the spec progresses.\ Samples are divided into 3 sub directories:

* [**Common**](/samples/python/common)  
Common code that all sample agents and apps use to speak A2A over HTTP. 

* [**Agents**](/samples/python/agents/README.md)  
Sample agents written in multiple frameworks that perform example tasks with tools. These all use the common A2AServer.

* [**Hosts**](/samples/python/hosts/README.md)  
Host applications that use the A2AClient. Includes a CLI which shows simple task completion with a single agent, a mesop web application that can speak to multiple agents, and an orchestrator agent that delegates tasks to one of multiple remote A2A agents.

## Prerequisites

- Python 3.13 or higher
- UV

## Running the Samples

Run one (or more) [agent](/samples/python/agents/README.md) A2A server and one of the [host applications](/samples/python/hosts/README.md). 

The following example will run the langgraph agent with the python CLI host:

1. Navigate to the samples/python directory:
    ```bash
    cd samples/python
    ```
2. Run an agent:
    ```bash
    uv run agents/langgraph
    ```
3. Run the example client
    ```
    uv run hosts/cli
    ```
---
**NOTE:** 
This is sample code and not production-quality libraries.
---
