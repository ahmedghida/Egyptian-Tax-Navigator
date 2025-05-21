# 🧾 Egyptian-Tax-Navigator – المساعد الضريبي الذكي لمصر

**Egyptian-Tax-Navigator** is an intelligent Arabic Q\&A assistant designed to answer questions related to **Egyptian tax law** using official legal documents only. It uses OCR to extract text from scanned PDFs, embeds the content into a **Chroma vector store**, and allows users to ask natural language questions in Arabic via a simple Streamlit web interface.

Built using **LangChain**, **Gemini 2.5**, **ChromaDB**, and **HuggingFace multilingual embeddings**.

---

## 📁 Project Structure

```
Egyptian-Tax-Navigator/
├── app.py                   # Streamlit interface
├── qa_chain.py             # Retrieval-Augmented Generation pipeline
├── create_pdf_vector_db.py # OCR + ChromaDB vectorization
├── assets/                 # Place tax law PDFs here
├── data/chroma_db/         # Vector DB (auto-generated)
├── .env                    # API key and configuration
└── requirements.txt        # All dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Egyptian-Tax-Navigator.git
cd Egyptian-Tax-Navigator
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Configure Environment Variables

Create a `.env` file in the root directory and add the following:

```env
GOOGLE_API_KEY="your_google_gemini_api_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token"
```

These keys are used for:

* 🔑 `GOOGLE_API_KEY`: Accessing **Gemini 2.5** via `langchain-google-genai`
* 🔑 `HUGGINGFACEHUB_API_TOKEN`: Loading **multilingual-e5-base** embeddings via `langchain-huggingface`

If you don’t already have these:

* [Get your Gemini API key](https://makersuite.google.com/app/apikey)
* [Get your Hugging Face token](https://huggingface.co/settings/tokens)

---



## 📚 Step 1: Add Tax PDFs

Place all your Egyptian tax-related PDFs (even scanned/image-based ones) in the `assets/` folder.

---

## 🧠 Step 2: Build the Vector Store with OCR

Run the script below to extract text and create the vector DB:

```bash
python create_pdf_vector_db.py
```

✅ This script:

* Uses Gemini 2.5 to extract text from each PDF page (even scanned images)
* Splits the content into chunks
* Saves it to **ChromaDB** using **intfloat/multilingual-e5-base** embeddings

---

## 🧾 Step 3: Launch the Assistant UI

```bash
streamlit run app.py
```

You’ll be able to:

* Type your question in Arabic (Egyptian dialect is fine)
* Get accurate responses based only on legal documents
* See `"لا أعلم"` if the system doesn't have enough context

---

## 🧠 How It Works

### 🧾 `create_pdf_vector_db.py`

* Extracts Arabic text from PDFs using **Gemini 2.5 Flash**
* Cleans and chunks it
* Stores embeddings in **Chroma vector store**

### 🔗 `qa_chain.py`

* Retrieves relevant chunks using semantic search
* Formats a strict Arabic-only prompt
* Uses Gemini to answer based on **retrieved context only**

### 🌐 `app.py`

* Streamlit UI for typing questions and receiving answers in real-time

---

## 🧪 Example Questions

* ❓ ما هي ضريبة القيمة المضافة على الخدمات الطبية؟
* ❓ كيف يتم حساب الضريبة على أرباح الشركات؟
* ❓ هل يُعفى الموظف من الضريبة على الدخل؟

---

## ❗ Bot Behavior Rules

* ✅ Always answers in Arabic (even Egyptian dialect)
* 🚫 Never hallucinates or guesses
* ❌ Will reply **"لا أعلم"** if context is insufficient

---

## 💡 Future Enhancements

* PDF upload via UI
* Arabic speech-to-text integration
* Voice-based Q\&A (ASR + TTS)
* Export answers as PDF

---

## 👤 Author

**Ahmed** – AI/ML Engineer | NLP & Computer Vision | 🇪🇬
Made with ❤️ using LangChain × Gemini × Streamlit

---

## 📌 License

This project is licensed for educational and non-commercial use.

---
