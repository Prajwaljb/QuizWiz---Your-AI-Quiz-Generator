# QuizWiz – AI Quiz Generator from PDFs

QuizWiz is a **Flask-based AI application** that automatically generates quizzes from uploaded PDF documents. It uses **NLP techniques** to extract key concepts from text and create high-quality **Fill-in-the-Blank** and **Multiple Choice Questions**, exporting them as a downloadable PDF.

The project is designed for students and educators who want quick, structured assessments from study material.

---

## 🚀 Features

* Upload any **PDF document**
* Automatic text extraction and cleaning
* AI-driven question generation

  * Fill-in-the-Blank questions
  * Multiple Choice Questions (MCQs)
* Context-aware distractor generation
* Clean, formatted **quiz PDF export**
* Simple web interface using Flask

---

## 🧠 How It Works

1. User uploads a PDF file
2. Text is extracted and preprocessed
3. Key concepts are identified using **TF-IDF** and **sentence embeddings**
4. Important words are selected to form questions
5. Distractors are generated based on semantic similarity
6. The final quiz is compiled into a downloadable PDF

---

## 🛠️ Tech Stack

* **Python**
* **Flask** (Web framework)
* **NLTK** (Tokenization, POS tagging)
* **Sentence-Transformers** (Semantic embeddings)
* **Scikit-learn** (TF-IDF)
* **PyPDF2** (PDF text extraction)
* **FPDF** (PDF generation)

---

## 📂 Project Structure

```
.
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
├── uploads/            # Temporary upload directory
├── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Prajwaljb/QuizWiz---Your-AI-Quiz-Generator.git
cd QuizWiz---Your-AI-Quiz-Generator
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\\Scripts\\activate    # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python app.py
```

Then open your browser and visit:

```
http://localhost:5000
```

---

## 📄 Output

* Automatically generated quiz
* Includes:

  * Fill-in-the-Blank section
  * Multiple Choice section
* Exported as a **PDF file** with answers included

---

## ⚠️ Notes

* Only **PDF files** are supported
* Large PDFs may take longer to process
* Internet access may be required on first run to download NLP models

---

## 👥 Authors

* **Prajwal JB**
* **Aashita Narayanpur**

---

## 📜 License

This project is intended for educational and academic use.

Feel free to fork, explore, and build upon it.
