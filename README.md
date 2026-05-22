# LOCALMIND-CHAT-APP

## 📌 Project Introduction

LOCALMIND-CHAT-APP is a powerful, locally-hosted AI chat application built using Python and Gradio.  
This project enables seamless interaction with both local Large Language Models (via Ollama) and cloud-based AI models (via OpenAI) using text, images, audio, and PDF documents.

The application combines multimodal AI capabilities, Retrieval-Augmented Generation (RAG), local speech-to-text processing, and persistent chat history into a single modern interface.

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

## 4.1 Multimodal Interaction

Supports interaction using:

* Text Chat
* Image Uploads
* Audio Inputs
* PDF Documents

---

## 4.2 Local & Cloud Models

Allows users to switch between:

* Local AI models using Ollama
* Cloud AI models using OpenAI APIs

---

## 4.3 Retrieval-Augmented Generation (RAG)

Enables PDF-based question answering by:

* Extracting PDF content
* Chunking text
* Creating embeddings
* Performing similarity search using ChromaDB

---

## 4.4 Local Audio Transcription

Uses the Whisper model for:

* Speech-to-text conversion
* Local/private audio processing
* Audio normalization using FFmpeg

---

## 4.5 Persistent Chat History

Stores:

* Chat messages
* Images
* Audio files
* User configurations

using SQLite database storage.

---

## 4.6 Dynamic Configuration

Allows dynamic adjustment of:

* RAG chunk size
* Retrieval limits
* Chat memory settings

through the UI.

---

# 📁 Project Structure & Execution Flow

## 5.1 Front-End & Entry Point

### `app.py`

The main application file responsible for:

* Running the Gradio server
* Managing UI components
* Handling session states
* Routing user interactions

---

## 5.2 Core Handlers

### `chat_api_handler.py`

Handles:

* API communication
* Prompt formatting
* Image encoding
* Streaming responses

Supports:

* OllamaChatAPIHandler
* OpenAIChatAPIHandler

---

### `audio_handler.py`

Responsible for:

* Speech-to-text processing
* Audio normalization
* `.webm` → `.wav` conversion
* Whisper model integration

---

### `pdf_handler.py`

Handles:

* PDF text extraction
* Document chunking
* LangChain text splitting

---

### `vectordb_handler.py`

Manages:

* ChromaDB vector storage
* Embedding generation
* Similarity search for RAG pipeline

---

## 5.3 Data & Utility

### `database_operations.py`

Provides:

* Thread-safe SQLite operations
* Chat history persistence
* Configuration storage

---

### `utils.py`

Contains utility functions for:

* Config loading
* Timestamp formatting
* Model fetching
* Performance timing

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
git clone <your-repo-url>
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

# 💡 How It Works (Integration)

## 9.1 Text Chat Workflow

```text
User Input
   ↓
app.py
   ↓
database_operations.py
   ↓
chat_api_handler.py
   ↓
Local / Cloud Model
   ↓
Save Response
   ↓
UI Update
```

---

## 9.2 Image Processing Workflow

```text
Image Upload
   ↓
app.py
   ↓
Base64 Conversion
   ↓
chat_api_handler.py
   ↓
Vision Model
```

---

## 9.3 Audio Processing Workflow

```text
Audio Input
   ↓
audio_handler.py
   ↓
FFmpeg Processing
   ↓
Whisper Model
   ↓
Transcribed Text
```

---

## 9.4 PDF Chat Workflow

```text
PDF Upload
   ↓
pdf_handler.py
   ↓
Text Chunking
   ↓
vectordb_handler.py
   ↓
Vector Embeddings
   ↓
Similarity Search
   ↓
Context Injection
```

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

# 💡 Skills Demonstrated

* Python Development
* AI Application Development
* Large Language Model Integration
* Retrieval-Augmented Generation (RAG)
* Speech-to-Text Processing
* Vector Database Management
* API Integration
* SQLite Database Handling
* UI Development with Gradio
* Multimodal AI Systems
* Prompt Engineering
* Backend Development

--- 
