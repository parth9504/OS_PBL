# 🔊 Smart Voice Assistant & AI-Powered Process Scheduler

An innovative **Generative AI-powered educational tool** combining a smart voice assistant and an interactive **process scheduling simulator** — designed to make **Operating Systems concepts more engaging, accessible, and intelligent.**

Built with Python, integrated with **Gemini API**, and featuring a rich **Streamlit UI** to simulate and visualize scheduling algorithms, generate PowerPoint presentations from voice input, and assist students in learning core OS concepts hands-on.

## 🎯 Project Objectives

- Use **Generative AI (Gemini API)** to empower learners with dynamic, real-time assistance.
- Simulate core **CPU scheduling algorithms** through voice and interactive UI.
- Convert OS theory into **visual aids and presentations** to enhance comprehension.
- Build a modular platform combining OS concepts, voice automation, and AI-powered insights.

---

## ✨ Key Features

### 🎤 Smart Voice Assistant (Python)
- Built using `speech_recognition`, `pyttsx3`, `wikipedia`, and more.
- Handles tasks like:
  - Web search
  - App launch
  - Task and notification management
  - Creating presentations with Gemini

### 🧠 GenAI-Powered PowerPoint Generator
- Accepts **voice prompts or typed input**
- Uses the **Gemini API** to:
  - Understand the topic
  - Generate structured slide content
  - Auto-create a PowerPoint presentation (`.pptx`) using `python-pptx`
- Perfect for students who need fast, AI-assisted presentations on OS topics or general subjects

### 🖥️ Process Scheduling Simulator (Interactive UI)
- **Streamlit-based UI** to input:
  - Arrival time, burst time, I/O burst time
  - Scheduling algorithm of choice (FCFS, SJF, RR, Multilevel feedback queue)
- Visualized **Gantt chart with real-time execution timeline**
- Supports **Round Robin Time Quantum** input

![image](https://github.com/user-attachments/assets/79561ba8-bfe7-4eb5-a7a3-bfa39b632283)


### 🤖 GenAI-Enhanced Scheduling Intelligence
- **"Suggest Best Algorithm"** button uses Gemini API to analyze input and recommend the most efficient scheduling algorithm
- **"Suggest Time Quantum"** (for Round Robin) uses Gemini to optimize performance based on process characteristics

![gantt_chart_animation](https://github.com/user-attachments/assets/daaacab1-7efd-48df-8fd7-476b3026909e)


## 🔧 Tech Stack

- **Languages**: Python
- **Voice Assistant**: `pyttsx3`, `speech_recognition`, `wikipedia`, `webbrowser`, etc.
- **Presentation**: `python-pptx`, Gemini API
- **Scheduling Simulator**: Python + Streamlit
- **GenAI Integration**: [Gemini API](https://ai.google.dev)

