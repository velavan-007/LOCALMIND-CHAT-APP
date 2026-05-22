# LOCALMIND-CHAT-APP
A powerful, locally-hosted Localmind chat application built with Python and Gradio. This project allows you to seamlessly interact with both local Large Language Models (via Ollama) and cloud-based models (via OpenAI) using text, audio, images, and PDF documents.

## 🚀 Features

- *Multimodal Interaction*: Chat seamlessly using text, images, audio (microphone or uploaded files), and PDFs.
- *Local & Cloud Models*: Instantly switch between running models locally via [Ollama](https://ollama.com/) (privacy-first) or utilizing [OpenAI](https://openai.com/)'s powerful cloud APIs.
- *Retrieval-Augmented Generation (RAG)*: Upload PDF documents and ask questions about their content. Uses ChromaDB and LangChain for chunking and vector storage.
- *Local Audio Transcription*: Uses a local deployment of the Whisper model (via HuggingFace transformers) for completely private speech-to-text transcription.
- *Persistent Chat History*: All conversations, media files, and settings are saved automatically in a SQLite database, allowing you to resume previous sessions anytime.
- *Dynamic Configuration*: Adjust RAG chunk size, document retrieval limits, and chat memory length directly from the UI.

## 📁 Project Structure & Execution Flow

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

## ⚙️ Configuration

The project uses a configuration file config.yaml to manage endpoints and model choices:

yaml
ollama:
  embedding_model: "nomic-embed-text"
  base_url: http://localhost:11434 # Use your local or Docker Ollama endpoint here
whisper_model: "openai/whisper-small" 

chromadb:
  chromadb_path: "chroma_db"
  collection_name: "pdfs"

chat_sessions_database_path: "./chat_sessions/chat_sessions.db"


## 🛠️ Installation & Setup

### Prerequisites
1. *Python 3.10+* installed on your system.
2. *FFmpeg*: Must be installed and added to your system PATH (required for audio transcription).
3. *Ollama*: (Optional but recommended) Install [Ollama](https://ollama.com/) to run models completely locally.

### Step 1: Clone the Repository
bash
git clone <your-repo-url>
cd LocalMind-AI-Chat


### Step 2: Install Dependencies
Create a virtual environment and install the required Python packages:
bash
python -m venv .venv
On Windows:
.venv\Scripts\activate
On Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt


### Step 3: API Keys (OpenAI)
If you plan to use OpenAI models in addition to local Ollama models, create a .env file in the root directory and add your API key:
env
OPENAI_API_KEY=your_openai_api_key_here


### Step 4: Setup Local Models (Ollama)
If you are using Ollama, ensure it is running in the background. You will need to pull the models you wish to use, as well as the embedding model for PDF Chat:
bash
ollama pull llama3
ollama pull nomic-embed-text


## 🚀 Running the Application

Start the Gradio web server by running:
bash
python app.py

Once the server starts, open the provided local URL (usually http://127.0.0.1:7860/) in your browser to start chatting!

## 💡 How It Works (Integration)

1. *Text Chat*: User inputs text -> app.py loads recent chat history from database_operations.py -> routes to chat_api_handler.py -> queries local or cloud model -> saves to DB and updates UI.
2. *Image Input*: User uploads image -> app.py detects image -> converts to base64 -> chat_api_handler.py packages it for Vision-capable models.
3. *Audio Input*: User records audio -> audio_handler.py transcodes it via ffmpeg -> processes it through local Whisper model -> transcribed text is piped to the text chat workflow.
4. *PDF Chat*: User uploads PDF -> pdf_handler.py chunks it -> vectordb_handler.py embeds and stores it -> Future questions retrieve relevant chunks and inject them as context into the prompt in chat_api_handler.py.

# 📸 Project Screenshots

## 🖥️ Main Chat Interface

![Main Chat Interface](https://github.com/velavan-007/LOCALMIND-CHAT-APP/blob/main/Main%20Chat%20Interface.jpeg)

## 🎙️ Audio Interaction Feature

![Audio Interaction Feature](https://github.com/velavan-007/LOCALMIND-CHAT-APP/blob/main/Audio%20Interaction%20Feature.jpeg)
