# LOCALMIND-CHAT-APP

## 📌 Project Introduction

LOCALMIND-CHAT-APP is a powerful, locally-hosted AI chat application built using Python and Gradio.This project enables seamless interaction with both local Large Language Models (via Ollama) and cloud-based AI models (via OpenAI) using text, images, audio and PDF documents.The application combines multimodal AI capabilities, Retrieval-Augmented Generation (RAG), local speech-to-text processing, and persistent chat history into a single modern interface.

---

# 🎯 Project Objective

The primary objectives of this project are:

* To build a fully functional local AI chat application
* To support multimodal interactions including text, image, audio, and PDFs
* To integrate both local and cloud-based LLMs
* To implement Retrieval-Augmented Generation (RAG) for PDF-based conversations
* To provide private and secure AI interactions using locally hosted models
* To create a user-friendly and interactive chat experience

---

# 🛠️ Tools & Technologies Used

* Python
* Gradio
* Ollama
* OpenAI API
* ChromaDB
* LangChain
* Whisper Model
* SQLite
* FFmpeg
* HuggingFace Transformers

---

# 🚀 Features
- *Multimodal Interaction*: Chat seamlessly using text, images, audio (microphone or uploaded files), and PDFs.
- *Local & Cloud Models*: Instantly switch between running models locally via [Ollama](https://ollama.com/) (privacy-first) or utilizing [OpenAI](https://openai.com/)'s powerful cloud APIs.
- *Retrieval-Augmented Generation (RAG)*: Upload PDF documents and ask questions about their content. Uses ChromaDB and LangChain for chunking and vector storage.
- *Local Audio Transcription*: Uses a local deployment of the Whisper model (via HuggingFace transformers) for completely private speech-to-text transcription.
- *Persistent Chat History*: All conversations, media files, and settings are saved automatically in a SQLite database, allowing you to resume previous sessions anytime.
- *Dynamic Configuration*: Adjust RAG chunk size, document retrieval limits, and chat memory length directly from the UI.



---

# 📁 Project Structure & Execution Flow

### Front-End & Entry Point
- *app.py*: The main entry point of the application. It runs the Gradio server, initializes the UI components, and routes all user interactions to the appropriate backend handlers. It maintains session states and updates the UI dynamically based on the interaction type (Text, PDF, Audio, or Image).

### Core Handlers
- *chat_api_handler.py*: Manages the core API communication. Contains handler classes (OllamaChatAPIHandler and OpenAIChatAPIHandler) that format prompts, manage multimodal inputs (like base64 image encoding), and handle streaming/non-streaming responses.
- *audio_handler.py*: Responsible for speech-to-text. It captures audio input, normalizes it using ffmpeg and librosa (handling .webm to .wav conversions), and feeds it into the local whisper-small model.
- *pdf_handler.py*: Uses pypdfium2 to extract raw text from PDF files, and LangChain's RecursiveCharacterTextSplitter to break the document down into configurable chunks before passing them to the Vector DB.
- *vectordb_handler.py*: Manages the local vector database using ChromaDB. It uses Ollama's nomic-embed-text model to embed document chunks and provides similarity search functionality for the RAG pipeline.

### Data & Utility
- *database_operations.py*: A robust, thread-safe SQLite wrapper that persists chat histories (messages table, storing texts and blobs) and UI configurations (settings table) across sessions.
- *utils.py*: Contains helper functions for loading configurations, fetching available models from APIs, formatting timestamps, and timing function executions.


---

# ⚙️ Configuration

The application uses a `config.yaml` file for managing model configurations and database settings.

```yaml
ollama:
  embedding_model: "nomic-embed-text"
  base_url: http://localhost:11434

whisper_model: "openai/whisper-small"

chromadb:
  chromadb_path: "chroma_db"
  collection_name: "pdfs"

chat_sessions_database_path: "./chat_sessions/chat_sessions.db"
```

---

# 🛠️ Installation & Setup

## Prerequisites

1. Python 3.10+
2. FFmpeg installed and added to system PATH
3. Ollama (Optional but Recommended)

Install Ollama:  
https://ollama.com/

---

## Step 1: Clone the Repository

```bash
git clone <https://github.com/velavan-007/LOCALMIND-CHAT-APP/tree/main>
cd LocalMind-AI-Chat
```

---

## Step 2: Install Dependencies

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 3: Configure OpenAI API Key

Create a `.env` file in the root directory.

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Step 4: Setup Ollama Models

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

---

# 🚀 Running the Application

Start the Gradio server:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:7860/
```

---



---

# 📸 Project Screenshots

## 🖥️ Main Chat Interface

![Main Chat Interface](https://github.com/velavan-007/LOCALMIND-CHAT-APP/blob/main/Main%20Chat%20Interface.jpeg)

---

## 🎙️ Audio Interaction Feature

![Audio Interaction Feature](https://github.com/velavan-007/LOCALMIND-CHAT-APP/blob/main/Audio%20Interaction%20Feature.jpeg)

---

# 🔍 Key Highlights

* Supports local AI execution using Ollama
* Enables multimodal AI interactions
* Provides private speech-to-text transcription
* Implements Retrieval-Augmented Generation (RAG)
* Maintains persistent chat history
* Offers dynamic UI-based configuration

---

# 🚀 Project Outcome

This project demonstrates the implementation of a modern multimodal AI assistant capable of handling text, images, audio, and PDF-based interactions using both local and cloud AI models.

The application provides an efficient, private, and scalable AI chat experience with advanced features like RAG, speech recognition, and persistent session management.

---


--- 
