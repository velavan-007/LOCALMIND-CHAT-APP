import gradio as gr
from chat_api_handler import ChatAPIHandler
from utils import get_timestamp, load_config
from audio_handler import transcribe_audio
from pdf_handler import add_documents_to_db
from database_operations import (
    get_db_manager,
    DEFAULT_CHAT_MEMORY_LENGTH,
    DEFAULT_RETRIEVED_DOCUMENTS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)
from utils import list_openai_models, list_ollama_models, command
import tempfile
import os

config = load_config()
db_manager = get_db_manager()

def get_session_history_ids():
    return ["new_session"] + db_manager.message_repo.get_all_chat_history_ids()

def create_temp_file(byte_data, ext):
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    with os.fdopen(fd, 'wb') as f:
        f.write(byte_data)
    return path

def format_history_for_gradio(session_id):
    if session_id == "new_session":
        return []
    messages = db_manager.message_repo.load_messages(session_id)
    history = []
    for msg in messages:
        role = msg["sender_type"]
        msg_type = msg["message_type"]
        content = msg["content"]
        if msg_type == 'text':
            history.append({"role": role, "content": content})
        elif msg_type == 'image':
            path = create_temp_file(content, 'jpg')
            history.append({"role": role, "content": {"path": path}})
        elif msg_type == 'audio':
            path = create_temp_file(content, 'wav')
            history.append({"role": role, "content": {"path": path}})
    return history

def update_model_options(api_choice):
    if api_choice == "ollama":
        opts = list_ollama_models()
        return gr.Dropdown(choices=opts, value=opts[0] if opts else None), gr.update(avatar_images=("chat_icons/USER.jfif", "chat_icons/OLLAMA.png"))
    else:
        opts = list_openai_models()
        return gr.Dropdown(choices=opts, value=opts[0] if opts else None), gr.update(avatar_images=("chat_icons/USER.jfif", "chat_icons/OPENAI.webp"))

def delete_chat(session_id): 
    if session_id != "new_session":
        db_manager.message_repo.delete_chat_history(session_id)
    new_choices = get_session_history_ids()
    return gr.Dropdown(choices=new_choices, value="new_session"), []

def save_setting(name, value):
    db_manager.settings_repo.update_setting(name, value)

def process_interaction(message_data, history, session_id, endpoint, model, pdf_chat, chat_memory_length, retrieved_docs, chunk_size, chunk_overlap):
    text = message_data.get("text", "")
    files = message_data.get("files", [])
    
    if session_id == "new_session":
        session_id = get_timestamp()
    
    pdf_files = [f for f in files if f.endswith('.pdf')]
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.ogg'))]
    
    history_for_ui = history.copy() if history else []
    
    # Process pdfs
    if pdf_files:
        pdf_bytes_list = []
        for pdf in pdf_files:
            with open(pdf, 'rb') as f:
                pdf_bytes_list.append(f.read())
        add_documents_to_db(pdf_bytes_list, chunk_size, chunk_overlap)
        
        if text:
            db_manager.message_repo.save_message(session_id, "user", "text", text)
            history_for_ui.append({"role": "user", "content": text})
            
            db_history = db_manager.message_repo.load_last_k_text_messages(session_id, chat_memory_length)
            formatted_db_history = [{"role": m["sender_type"], "content": m["content"]} for m in db_history]
            llm_answer = ChatAPIHandler.chat(user_input=text, chat_history=formatted_db_history, 
                                             endpoint_to_use=endpoint, model_to_use=model, 
                                             pdf_chat=pdf_chat, retrieved_documents=retrieved_docs)
            db_manager.message_repo.save_message(session_id, "assistant", "text", llm_answer)
            history_for_ui.append({"role": "assistant", "content": llm_answer})
            
    # Process images
    if image_files:
        for img_path in image_files:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            history_for_ui.append({"role": "user", "content": {"path": img_path}})
            db_manager.message_repo.save_message(session_id, "user", "image", img_bytes)
            
            db_manager.message_repo.save_message(session_id, "user", "text", text)
            if text:
                history_for_ui.append({"role": "user", "content": text})
            
            llm_answer = ChatAPIHandler.chat(user_input=text, chat_history=[], 
                                             endpoint_to_use=endpoint, model_to_use=model, 
                                             pdf_chat=False, retrieved_documents=retrieved_docs, image=img_bytes)
            db_manager.message_repo.save_message(session_id, "assistant", "text", llm_answer)
            history_for_ui.append({"role": "assistant", "content": llm_answer})
            
    # Process audio
    if audio_files:
        for audio_path in audio_files:
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            transcribed = transcribe_audio(audio_bytes)
            combined_text = (text + "\n" + transcribed).strip()
            
            history_for_ui.append({"role": "user", "content": {"path": audio_path}})
            
            db_manager.message_repo.save_message(session_id, "user", "audio", audio_bytes)
            db_manager.message_repo.save_message(session_id, "user", "text", combined_text)
            
            llm_answer = ChatAPIHandler.chat(user_input=combined_text, chat_history=[], 
                                             endpoint_to_use=endpoint, model_to_use=model, 
                                             pdf_chat=pdf_chat, retrieved_documents=retrieved_docs)
            db_manager.message_repo.save_message(session_id, "assistant", "text", llm_answer)
            history_for_ui.append({"role": "assistant", "content": llm_answer})
            
    # Just text
    if text and not (pdf_files or image_files or audio_files):
        if text.startswith("/"):
            response = command(text)
            db_manager.message_repo.save_message(session_id, "user", "text", text)
            db_manager.message_repo.save_message(session_id, "assistant", "text", str(response))
            history_for_ui.append({"role": "user", "content": text})
            history_for_ui.append({"role": "assistant", "content": str(response)})
        else:
            db_manager.message_repo.save_message(session_id, "user", "text", text)
            history_for_ui.append({"role": "user", "content": text})
            db_history = db_manager.message_repo.load_last_k_text_messages(session_id, chat_memory_length)
            formatted_db_history = [{"role": m["sender_type"], "content": m["content"]} for m in db_history[:-1]] # exclude the one we just saved
            llm_answer = ChatAPIHandler.chat(user_input=text, chat_history=formatted_db_history, 
                                             endpoint_to_use=endpoint, model_to_use=model, 
                                             pdf_chat=pdf_chat, retrieved_documents=retrieved_docs)
            db_manager.message_repo.save_message(session_id, "assistant", "text", llm_answer)
            history_for_ui.append({"role": "assistant", "content": llm_answer})

    sessions = get_session_history_ids()
    new_dd = gr.Dropdown(choices=sessions, value=session_id)
    
    return history_for_ui, new_dd, session_id, {"text": "", "files": []}

def process_audio_mic(audio_filepath, history, session_id, endpoint, model, pdf_chat, chat_memory_length, retrieved_docs):
    if not audio_filepath:
        return history, gr.Dropdown(), session_id, None
        
    if session_id == "new_session":
        session_id = get_timestamp()
        
    with open(audio_filepath, 'rb') as f:
        audio_bytes = f.read()
    transcribed = transcribe_audio(audio_bytes)
    
    history_for_ui = history.copy() if history else []
    history_for_ui.append({"role": "user", "content": {"path": audio_filepath}})
    history_for_ui.append({"role": "user", "content": transcribed})
    
    db_manager.message_repo.save_message(session_id, "user", "audio", audio_bytes)
    db_manager.message_repo.save_message(session_id, "user", "text", transcribed)

    db_history = db_manager.message_repo.load_last_k_text_messages(session_id, chat_memory_length)
    formatted_db_history = [{"role": m["sender_type"], "content": m["content"]} for m in db_history[:-1]]
    
    llm_answer = ChatAPIHandler.chat(user_input=transcribed, chat_history=formatted_db_history, 
                                     endpoint_to_use=endpoint, model_to_use=model, 
                                     pdf_chat=pdf_chat, retrieved_documents=retrieved_docs)
    db_manager.message_repo.save_message(session_id, "assistant", "text", llm_answer)
    history_for_ui.append({"role": "assistant", "content": llm_answer})
    
    sessions = get_session_history_ids()
    new_dd = gr.Dropdown(choices=sessions, value=session_id)
    
    return history_for_ui, new_dd, session_id, None


with gr.Blocks(title="LocalMind Chat App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("LocalMind Chat App")
    
    current_session = gr.State("new_session")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Chat Sessions")
            session_dropdown = gr.Dropdown(choices=get_session_history_ids(), value="new_session", label="Select a chat session", interactive=True)
            
            gr.Markdown("### Configuration")
            api_dropdown = gr.Dropdown(choices=["ollama", "openai"], value="ollama", label="Select an API")
            
            initial_models = list_ollama_models()
            model_dropdown = gr.Dropdown(choices=initial_models, value=initial_models[0] if initial_models else None, label="Select a Model")
            
            pdf_toggle = gr.Checkbox(label="PDF Chat", value=False)
            
            mic_audio = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio")
            
            delete_btn = gr.Button("Delete Chat Session", variant="stop")
            
            gr.Markdown("### Chat History")
            chat_memory_slider = gr.Number(
                value=int(db_manager.settings_repo.get_setting("chat_memory_length", DEFAULT_CHAT_MEMORY_LENGTH)), 
                label="Number of Previous Messages"
            )
            chat_memory_slider.change(fn=lambda x: save_setting("chat_memory_length", x), inputs=[chat_memory_slider])
            
            gr.Markdown("### PDF Processing")
            retrieved_docs_input = gr.Number(
                value=int(db_manager.settings_repo.get_setting("retrieved_documents", DEFAULT_RETRIEVED_DOCUMENTS)),
                label="Number of Retrieved PDF Chunks"
            )
            retrieved_docs_input.change(fn=lambda x: save_setting("retrieved_documents", x), inputs=[retrieved_docs_input])
            
            chunk_size_input = gr.Number(
                value=int(db_manager.settings_repo.get_setting("chunk_size", DEFAULT_CHUNK_SIZE)),
                label="PDF Chunk Size (characters)"
            )
            chunk_size_input.change(fn=lambda x: save_setting("chunk_size", x), inputs=[chunk_size_input])
            
            chunk_overlap_input = gr.Number(
                value=int(db_manager.settings_repo.get_setting("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
                label="PDF Chunk Overlap (characters)"
            )
            chunk_overlap_input.change(fn=lambda x: save_setting("chunk_overlap", x), inputs=[chunk_overlap_input])
            
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600, type="messages", avatar_images=("chat_icons/USER.jfif", "chat_icons/OLLAMA.png"))
            msg_input = gr.MultimodalTextbox(
                file_types=["image", "audio", ".pdf"],
                placeholder="Type your message here...",
                interactive=True,
                show_label=False
            )
            
            # Event listeners
            api_dropdown.change(fn=update_model_options, inputs=[api_dropdown], outputs=[model_dropdown, chatbot])
            
            session_dropdown.change(fn=format_history_for_gradio, inputs=[session_dropdown], outputs=[chatbot]).then(
                fn=lambda id: id, inputs=[session_dropdown], outputs=[current_session]
            )
            
            delete_btn.click(fn=delete_chat, inputs=[current_session], outputs=[session_dropdown, chatbot]).then(
                fn=lambda id: id, inputs=[session_dropdown], outputs=[current_session]
            )
            
            msg_input.submit(
                fn=process_interaction,
                inputs=[
                    msg_input, chatbot, current_session, api_dropdown, model_dropdown, 
                    pdf_toggle, chat_memory_slider, retrieved_docs_input, chunk_size_input, chunk_overlap_input
                ],
                outputs=[chatbot, session_dropdown, current_session, msg_input]
            )
            
            mic_audio.stop_recording(
                fn=process_audio_mic,
                inputs=[
                    mic_audio, chatbot, current_session, api_dropdown, model_dropdown,
                    pdf_toggle, chat_memory_slider, retrieved_docs_input
                ],
                outputs=[chatbot, session_dropdown, current_session, mic_audio]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
