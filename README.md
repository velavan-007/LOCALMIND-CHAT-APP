````md
# LOCALMIND-CHAT-APP

A powerful, locally-hosted LocalMind chat application built with Python and Gradio.  
This project allows you to seamlessly interact with both local Large Language Models (via Ollama) and cloud-based models (via OpenAI) using text, audio, images, and PDF documents.

---

# 🚀 Features

- **Multimodal Interaction**  
  Chat seamlessly using text, images, audio (microphone or uploaded files), and PDFs.

- **Local & Cloud Models**  
  Instantly switch between running models locally via [Ollama](https://ollama.com/) (privacy-first) or utilizing [OpenAI](https://openai.com/) powerful cloud APIs.

- **Retrieval-Augmented Generation (RAG)**  
  Upload PDF documents and ask questions about their content. Uses ChromaDB and LangChain for chunking and vector storage.

- **Local Audio Transcription**  
  Uses a local deployment of the Whisper model (via HuggingFace Transformers) for completely private speech-to-text transcription.

- **Persistent Chat History**  
  All conversations, media files, and settings are saved automatically in a SQLite database, allowing you to resume previous sessions anytime.

- **Dynamic Configuration**  
  Adjust RAG chunk size, document retrieval limits, and chat memory length directly from the UI.

---

# 📁 Project Structure & Execution Flow

## Front-End & Entry Point

### `app.py`
The main entry point of the application.  
It runs the Gradio server, initializes the UI components, and routes all user interactions to the appropriate backend handlers.  
It maintains session states and updates the UI dynamically based on the interaction type (Text, PDF, Audio, or Image).

---

## Core Handlers

### `chat_api_handler.py`
Manages the core API communication.

- Contains handler classes:
  - `OllamaChatAPIHandler`
  - `OpenAIChatAPIHandler`
- Formats prompts
- Manages multimodal inputs (like Base64 image encoding)
- Handles streaming and non-streaming responses

### `audio_handler.py`
Responsible for speech-to-text processing.

- Captures audio input
- Normalizes audio using `ffmpeg` and `librosa`
- Handles `.webm` → `.wav` conversion
- Processes audio using the local `whisper-small` model

### `pdf_handler.py`
Handles PDF processing.

- Extracts raw text using `pypdfium2`
- Splits content using LangChain's `RecursiveCharacterTextSplitter`
- Creates configurable document chunks for vector storage

### `vectordb_handler.py`
Manages the local vector database using ChromaDB.

- Uses Ollama's `nomic-embed-text` model for embeddings
- Stores document chunks
- Provides similarity search for the RAG pipeline

---

## Data & Utility

### `database_operations.py`
A robust, thread-safe SQLite wrapper that:

- Persists chat histories
- Stores text and media blobs
- Saves UI configurations and settings

### `utils.py`
Contains utility/helper functions for:

- Loading configurations
- Fetching available models from APIs
- Formatting timestamps
- Timing function executions

---

# ⚙️ Configuration

The project uses a `config.yaml` file to manage endpoints and model settings.

```yaml
ollama:
  embedding_model: "nomic-embed-text"
  base_url: http://localhost:11434

whisper_model: "openai/whisper-small"

chromadb:
  chromadb_path: "chroma_db"
  collection_name: "pdfs"

chat_sessions_database_path: "./chat_sessions/chat_sessions.db"
````

---

# 🛠️ Installation & Setup

## Prerequisites

1. **Python 3.10+** installed on your system
2. **FFmpeg** installed and added to system PATH
3. **Ollama** (optional but recommended)

Install Ollama:
[https://ollama.com/](https://ollama.com/)

---

## Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd LocalMind-AI-Chat
```

---

## Step 2: Install Dependencies

Create a virtual environment and install dependencies.

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

## Step 3: Setup OpenAI API Key

Create a `.env` file in the root directory.

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Step 4: Setup Local Models (Ollama)

Ensure Ollama is running in the background.

Pull the required models:

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

Once the server starts, open:

```text
http://127.0.0.1:7860/
```

in your browser to start chatting.

---

# 💡 How It Works (Integration)

## 1. Text Chat

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
Save to Database
   ↓
Update UI
```

---

## 2. Image Input

```text
User Uploads Image
   ↓
app.py Detects Image
   ↓
Convert to Base64
   ↓
chat_api_handler.py
   ↓
Vision-Capable Model
```

---

## 3. Audio Input

```text
User Records Audio
   ↓
audio_handler.py
   ↓
ffmpeg Processing
   ↓
Whisper Model
   ↓
Transcribed Text
   ↓
Text Chat Workflow
```

---

## 4. PDF Chat

```text
User Uploads PDF
   ↓
pdf_handler.py
   ↓
Chunk Creation
   ↓
vectordb_handler.py
   ↓
Vector Embeddings
   ↓
Similarity Search
   ↓
Context Injection into Prompt
```

---

# 📸 Project Screenshots

## 🖥️ Main Chat Interface

![Main Chat Interface](https://github.com/velavan-007/LOCALMIND-CHAT-APP/blob/main/Main%20Chat%20Interface.jpeg)

---

## 🎙️ Audio Interaction Feature

![Audio Interaction Feature](https://github.com/velavan-007/LOCALMIND-CHAT-APP/blob/main/Audio%20Interaction%20Feature.jpeg)

```
```
