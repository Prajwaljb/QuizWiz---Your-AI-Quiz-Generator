from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import random
from PyPDF2 import PdfReader
from fpdf import FPDF
import unicodedata
import logging
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NLP components
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Load lightweight sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model: {e}")
    raise RuntimeError("Could not initialize NLP model")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return None

def clean_text(text):
    """Enhanced cleaning for quiz generation"""
    # Fix hyphenated words
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    # Fix broken words
    text = re.sub(r'(\w)\s*-\s*(\w)', r'\1\2', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove citation numbers
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

def extract_key_concepts(text, num_concepts=15):
    """Extract key concepts using TF-IDF and embeddings"""
    sentences = [s for s in sent_tokenize(text) if len(s) > 20]
    
    # TF-IDF approach
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top terms
    tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    top_tfidf_indices = tfidf_scores.argsort()[::-1][:num_concepts*2]
    tfidf_terms = [feature_names[i] for i in top_tfidf_indices]
    
    # Embedding-based approach
    sentence_embeddings = model.encode(sentences)
    doc_embedding = np.mean(sentence_embeddings, axis=0)
    
    # Score terms by embedding similarity to document
    term_scores = {}
    for term in set(tfidf_terms):
        term_embedding = model.encode(term)
        similarity = np.dot(term_embedding, doc_embedding) / (
            np.linalg.norm(term_embedding) * np.linalg.norm(doc_embedding))
        term_scores[term] = similarity
    
    # Combine scores and return top terms
    sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in sorted_terms[:num_concepts]]

def is_quality_sentence(sentence):
    """Stricter validation for sentences"""
    if len(sentence) < 25 or len(sentence) > 200:
        return False
    if sentence.count(' ') < 5:  # Too short
        return False
    if re.search(r'\w-\s*\w', sentence):  # Reject broken words
        return False
    if any(char.isdigit() for char in sentence):  # Reject sentences with numbers
        return False
    return True

def find_best_blank(sentence, key_concepts):
    """Find better words to blank out"""
    words = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    
    # Prefer words that:
    # 1. Are in key concepts
    # 2. Are nouns/verbs
    # 3. Are between 4-12 characters
    candidates = []
    for word, pos in pos_tags:
        if (4 <= len(word) <= 12 and
            word.isalpha() and
            pos.startswith(('NN', 'VB', 'JJ'))):
            score = 1
            if word.lower() in [c.lower() for c in key_concepts]:
                score += 2
            candidates.append((word, score))
    
    if not candidates:
        return None
    
    # Return highest scoring candidate
    return max(candidates, key=lambda x: x[1])[0]

def generate_fill_blank_questions(text, num_questions=10):
    """Generate fill-in-the-blank questions"""
    key_concepts = extract_key_concepts(text)
    sentences = [s for s in sent_tokenize(text) if is_quality_sentence(s)]
    questions = []
    used_words = set()
    
    for sentence in sentences:
        if len(questions) >= num_questions:
            break
            
        blank_word = find_best_blank(sentence, key_concepts)
        if blank_word and blank_word.lower() not in used_words:
            question = sentence.replace(blank_word, '_____', 1)
            question = re.sub(r'\s+', ' ', question).strip()
            
            if (10 <= len(question) <= 120 and 
                question.count('_____') == 1 and
                not question.startswith('_____') and
                not question.endswith('_____')):
                
                questions.append((question, blank_word))
                used_words.add(blank_word.lower())
    
    return questions[:num_questions]

def generate_distractors(target_word, sentence, key_concepts, num_distractors=3):
    """Generate context-aware distractors"""
    # Get similar words from key concepts
    distractors = []
    target_lower = target_word.lower()
    
    for concept in key_concepts:
        concept_lower = concept.lower()
        if (concept_lower != target_lower and 
            len(concept) > 3 and
            concept.isalpha() and
            concept_lower not in stopwords.words('english')):
            distractors.append(concept)
            if len(distractors) >= num_distractors*2:
                break
    
    # If not enough, use words from the same sentence
    if len(distractors) < num_distractors:
        words = word_tokenize(sentence)
        for word in words:
            word_lower = word.lower()
            if (word_lower != target_lower and 
                len(word) > 3 and
                word.isalpha() and
                word_lower not in stopwords.words('english') and
                word not in distractors):
                distractors.append(word)
                if len(distractors) >= num_distractors*2:
                    break
    
    # Select top distractors by length similarity
    distractors = sorted(distractors, 
                        key=lambda x: abs(len(x) - len(target_word)))[:num_distractors]
    return distractors

def generate_mcq_questions(text, num_questions=10):
    key_concepts = extract_key_concepts(text)
    sentences = [s for s in sent_tokenize(text) if is_quality_sentence(s)]
    mcqs = []
    
    for sentence in sentences:
        if len(mcqs) >= num_questions:
            break
            
        target_word = find_best_blank(sentence, key_concepts)
        if target_word and len(target_word) > 3:
            distractors = generate_distractors(target_word, sentence, key_concepts)
            
            if len(distractors) >= 2:  # Reduced from 3 to get more questions
                question = ' '.join(sentence.replace(target_word, '_____').split())
                options = distractors[:2] + [target_word]
                random.shuffle(options)
                
                mcqs.append((question, options, target_word))
    
    return mcqs[:num_questions]

def write_wrapped_text(pdf, text, indent=0, max_width=150):
    """Improved text wrapping that prevents word breaks"""
    text = str(text).strip()
    pdf.set_x(indent + pdf.l_margin)
    
    # First split into paragraphs
    paragraphs = text.split('\n')
    for para in paragraphs:
        words = para.split()
        line = ""
        
        for word in words:
            # Never break words, just move to next line
            test_line = f"{line} {word}" if line else word
            if pdf.get_string_width(test_line) <= max_width:
                line = test_line
            else:
                if line:
                    pdf.cell(0, 8, line, ln=True)
                    pdf.set_x(indent + pdf.l_margin)
                line = word
        
        if line:
            pdf.cell(0, 8, line, ln=True)
            pdf.set_x(indent + pdf.l_margin)

# In generate_pdf() function, modify these sections:
def generate_pdf(fib_questions, mcq_questions, filename="quiz_output.pdf"):
    pdf = FPDF()
    pdf.set_margins(25, 25, 25)
    pdf.set_auto_page_break(True, margin=25)
    
    # Header
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, "KNOWLEDGE ASSESSMENT", 0, 1, 'C')
    pdf.ln(10)
    
    # Fill-in-blank section
    if fib_questions:
        pdf.set_font("helvetica", 'B', 14)
        pdf.cell(0, 10, "Part 1: Fill in the Blanks", 0, 1)
        pdf.set_font("helvetica", size=12)
        
        for i, (q, a) in enumerate(fib_questions, 1):
            # Clean question text
            clean_q = ' '.join(q.split())
            pdf.multi_cell(0, 8, f"{i}. {clean_q}", 0, 1)
            pdf.set_font("helvetica", 'I', 10)
            pdf.cell(0, 8, f"Answer: {a}", 0, 1)
            pdf.set_font("helvetica", size=12)
            pdf.ln(5)
    
    # Only add new page if we have MCQs
    if mcq_questions:
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 14)
        pdf.cell(0, 10, "Part 2: Multiple Choice", 0, 1)
        pdf.set_font("helvetica", size=12)
        
        for i, (q, options, a) in enumerate(mcq_questions, 1):
            clean_q = ' '.join(q.split())
            pdf.multi_cell(0, 8, f"{i}. {clean_q}", 0, 1)
            
            for opt_num, option in enumerate(options, 1):
                clean_opt = ' '.join(str(option).split())
                pdf.cell(20)  # Indent
                pdf.multi_cell(0, 8, f"{chr(96+opt_num)}. {clean_opt}", 0, 1)
            
            pdf.set_font("helvetica", 'I', 10)
            pdf.cell(0, 8, f"Correct answer: {a}", 0, 1)
            pdf.set_font("helvetica", size=12)
            pdf.ln(8)
    
    # Save PDF
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.output(output_path)
    return output_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload a PDF file.', 'error')
                return redirect(request.url)
                
            # Process file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get parameters with validation
            try:
                fib_count = int(request.form.get('fib_count', 10))
                fib_count = max(1, min(fib_count, 20))
                
                mcq_count = int(request.form.get('mcq_count', 10))
                mcq_count = max(1, min(mcq_count, 20))
            except ValueError:
                flash('Invalid question count values', 'error')
                return redirect(request.url)
            
            # Extract and process text
            raw_text = extract_text_from_pdf(filepath)
            if not raw_text or len(raw_text) < 200:
                flash('The PDF contains too little text to generate quality questions', 'error')
                return redirect(request.url)
                
            cleaned_text = clean_text(raw_text)
            
            # Generate questions
            fib_questions = generate_fill_blank_questions(cleaned_text, fib_count)
            mcq_questions = generate_mcq_questions(cleaned_text, mcq_count)
            
            if not fib_questions and not mcq_questions:
                flash('Could not generate questions from this content. Try a different document.', 'error')
                return redirect(request.url)
                
            # Export PDF
            try:
                output_path = generate_pdf(fib_questions, mcq_questions)
                return send_file(
                    output_path,
                    as_attachment=True,
                    download_name="knowledge_quiz.pdf",
                    mimetype='application/pdf'
                )
            except Exception as e:
                flash("Error generating PDF. Please try with different content.")
                logger.error(f"PDF generation failed: {str(e)}")
                return redirect(request.url)
            finally:
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)