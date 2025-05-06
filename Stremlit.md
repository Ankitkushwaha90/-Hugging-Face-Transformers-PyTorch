Awesome! Here's how you can create a 🖥️ Web Interface (using Streamlit) for the PDF Summarizer app using OpenAI, LangChain, and PyPDF.

### 🧰 Requirements
📦 Install Required Libraries
```bash
pip install openai langchain pypdf streamlit
```
### 📁 Project Structure
```bash
pdf_summarizer_app/
├── app.py
├── sample.pdf
```
### 🧾 app.py – Streamlit PDF Summarizer UI
```python
import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from pypdf import PdfReader

# Function to extract text
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Summarize using LangChain
def summarize_pdf(text, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    docs = [Document(page_content=text)]
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.run(docs)

# Streamlit UI
st.set_page_config(page_title="📄 PDF Summarizer", layout="centered")
st.title("📄 PDF Summarizer using LangChain & OpenAI")

api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
pdf_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if st.button("🧠 Summarize") and pdf_file and api_key:
    with st.spinner("Reading PDF and summarizing..."):
        raw_text = extract_text(pdf_file)
        summary = summarize_pdf(raw_text, api_key)
        st.success("✅ Done!")
        st.subheader("📝 Summary:")
        st.write(summary)
```
### ▶️ How to Run the App
In your terminal:

```bash
streamlit run app.py
```
Then open your browser to http://localhost:8501

### 🧠 What You Can Add Next:
- PDF chunking for large documents

- File download button for summary

- Add LLM Q&A functionality

- Optional chatbot below the summary

- Token counter using tiktoken

Would you like to add PDF Q&A chat next or deploy this on the web (e.g. with Hugging Face or Render)?
