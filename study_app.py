import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import fitz  # PyMuPDF

# Load model
model = T5ForConditionalGeneration.from_pretrained('./t5_finetuned')
tokenizer = T5Tokenizer.from_pretrained('./t5_finetuned')

memory = {"pdf": "", "notes": ""}

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    memory["pdf"] = text
    return "✅ PDF uploaded successfully!"

# Upload notes
def upload_notes(text):
    memory["notes"] = text
    return "✅ Notes uploaded successfully!"

# Summarize content
def summarize(source):
    content = memory[source]
    if not content:
        return "❗ No content uploaded yet."
    input_text = f"summarize: {content}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Ask questions
def ask_question(user_input, chat_history, source):
    context = memory[source]
    if not context:
        chat_history.append({"role": "assistant", "content": "❗ Please upload content first."})
        return chat_history
    input_text = f"question: {user_input} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📚 AI Study Assistant")

    with gr.Tab("📄 Upload PDF"):
        with gr.Column():
            gr.Markdown("### Step 1: Upload PDF")
            pdf_file = gr.File(label="Upload PDF Lecture", type="filepath")
            upload_pdf_btn = gr.Button("Upload PDF")
            pdf_status = gr.Textbox(label="", interactive=False)

            gr.Markdown("### Step 2: Summarize PDF")
            summarize_pdf_btn = gr.Button("Summarize PDF")
            pdf_summary = gr.Textbox(label="PDF Summary", lines=6)

            gr.Markdown("### Step 3: Ask Questions")
            pdf_chatbot = gr.Chatbot(label="Chat with your PDF", type="messages")
            pdf_question = gr.Textbox(label="Your Question", placeholder="Ask something about the PDF...")
            pdf_ask_btn = gr.Button("Ask")

        upload_pdf_btn.click(fn=extract_text_from_pdf, inputs=pdf_file, outputs=pdf_status)
        summarize_pdf_btn.click(fn=lambda: summarize("pdf"), outputs=pdf_summary)
        pdf_ask_btn.click(fn=lambda q, h: ask_question(q, h, "pdf"), inputs=[pdf_question, pdf_chatbot], outputs=pdf_chatbot)

    with gr.Tab("📝 Upload Notes"):
        with gr.Column():
            gr.Markdown("### Step 1: Paste Notes")
            notes_input = gr.Textbox(label="Your Notes", lines=10, placeholder="Paste your notes here...")
            upload_notes_btn = gr.Button("Upload Notes")
            notes_status = gr.Textbox(label="", interactive=False)

            gr.Markdown("### Step 2: Summarize Notes")
            summarize_notes_btn = gr.Button("Summarize Notes")
            notes_summary = gr.Textbox(label="Notes Summary", lines=6)

            gr.Markdown("### Step 3: Ask Questions")
            notes_chatbot = gr.Chatbot(label="Chat with your Notes", type="messages")
            notes_question = gr.Textbox(label="Your Question", placeholder="Ask something about your notes...")
            notes_ask_btn = gr.Button("Ask")

        upload_notes_btn.click(fn=upload_notes, inputs=notes_input, outputs=notes_status)
        summarize_notes_btn.click(fn=lambda: summarize("notes"), outputs=notes_summary)
        notes_ask_btn.click(fn=lambda q, h: ask_question(q, h, "notes"), inputs=[notes_question, notes_chatbot], outputs=notes_chatbot)

demo.launch(share=True)