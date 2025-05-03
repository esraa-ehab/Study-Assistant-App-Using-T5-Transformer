# Study Assistant: Lecture Summarizer and Q&A System

This project leverages a fine-tuned **T5** model to summarize lecture transcripts and answer questions about the content. It uses Hugging Face's `transformers` library and works with a custom-trained model.

## Features

- **Lecture Summarization**: Given a lecture transcript, the model summarizes the key points in a concise format.
- **Question Answering**: The model can also answer questions based on the lecture content, providing detailed responses.
  
## Prerequisites

Before using the model, ensure you have the following installed:

- Python 3.6+
- `transformers` library
- `torch` (PyTorch)
- `Gradio` (for creating a web app)
- Any other dependencies (listed below)

### Installing Dependencies

You can install the necessary dependencies using `pip`. Below is a list of packages required:

```bash
pip install transformers torch gradio
