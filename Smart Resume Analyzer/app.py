import os
import json
import csv
import re
from datetime import timedelta, datetime
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import joblib
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
load_dotenv()
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your_super_super_secret_key_here')

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Failed to download NLTK data: {e}")
    exit(1)

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.permanent_session_lifetime = timedelta(minutes=30)

UPLOAD_FOLDER = 'static/resumes'
PARSED_RESUMES_JSON = 'parsed_resumes.json'
USERS_CSV = 'users.csv'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    model = joblib.load('rf_resume_model.pkl')
    vectorizer = joblib.load('rf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully")
except FileNotFoundError as e:
    print(f"Model or vectorizer file not found: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    exit(1)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

SKILLS = [
    "Python", "R", "SQL", "Machine Learning", "Deep Learning", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
    "Power BI", "Tableau", "Data Visualization", "Data Wrangling", "Statistics", "Big Data", "NLP",
    "Recruitment", "Employee Relations", "Talent Acquisition", "Performance Management", "HRIS", "Payroll",
    "Organizational Development", "Communication Skills", "Conflict Resolution", "Training & Development",
    "Legal Research", "Drafting", "Litigation", "Legal Compliance", "Case Analysis", "Negotiation", "Civil Law",
    "Criminal Law", "Contract Law", "Creativity", "Graphic Design", "Adobe Photoshop", "Illustrator", "Sketching",
    "Visual Storytelling", "Fine Arts", "Color Theory", "Art History", "Portfolio Management", "HTML", "CSS",
    "JavaScript", "UI/UX Design", "Adobe XD", "Figma", "Responsive Design", "Bootstrap", "Web Accessibility",
    "AutoCAD", "SolidWorks", "Ansys", "Mechanical Design", "Thermodynamics", "Fluid Mechanics", "MATLAB",
    "CNC Programming", "Problem-Solving", "Technical Drawing", "CRM Tools", "Customer Service", "Lead Generation",
    "Sales Strategy", "Market Analysis", "Product Knowledge", "Target Achievement", "Personal Training",
    "Nutrition", "Fitness Assessment", "Exercise Planning", "CPR & First Aid", "Coaching", "Wellness Programs",
    "Strength Training", "Flexibility Training", "STAAD Pro", "Project Management", "Structural Analysis",
    "Construction Management", "Surveying", "Estimation", "Site Supervision", "Safety Regulations", "Spring Boot",
    "Hibernate", "REST APIs", "OOP", "Maven", "Jenkins", "JUnit", "Git", "Microservices", "Multithreading",
    "Eclipse", "IntelliJ IDEA", "Requirement Gathering", "Documentation", "Agile", "JIRA", "Use Case Modeling",
    "BPMN", "Stakeholder Management", "MS Excel", "SAP ABAP", "SAP FICO", "SAP HANA", "SAP MM", "SAP SD", "BAPI",
    "BADI", "Data Migration", "SAP UI5", "SAP BASIS", "Functional Specifications", "Selenium", "TestNG",
    "Automation Frameworks", "API Testing", "Postman", "Cucumber", "Unit Testing", "CI/CD", "Circuit Design",
    "Power Systems", "PCB Design", "Electrical Maintenance", "SCADA", "PLC", "Control Systems", "Load Flow Analysis",
    "Wiring Diagrams", "Operations Management", "Supply Chain", "Strategic Planning", "Team Leadership",
    "Logistics", "Budgeting", "ERP Systems", "Lean Manufacturing", "Flask", "Web Scraping", "SQLite", "PostgreSQL",
    "Data Structures", "Docker", "Kubernetes", "Ansible", "Terraform", "AWS", "Azure", "Monitoring Tools",
    "Bash Scripting", "Shell Scripting", "Network Protocols", "Firewalls", "IDS", "IPS", "VPNs", "Cybersecurity",
    "Ethical Hacking", "Wireshark", "Network Monitoring", "ISO Standards", "Risk Assessment", "Project Planning",
    "MS Project", "Risk Management", "Reporting", "Project Lifecycle", "PMBOK", "Oracle", "MySQL", "MongoDB",
    "PL/SQL", "Data Modeling", "Backup & Recovery", "Database Administration", "Indexing", "Query Optimization",
    "HDFS", "MapReduce", "Hive", "Pig", "HBase", "Spark", "Sqoop", "Kafka", "YARN", "Cloudera", "Data Lake",
    "Informatica", "Talend", "Data Warehousing", "Data Mapping", "SSIS", "Data Quality", "Data Transformation",
    "DBMS", "C#", ".NET Framework", "ASP.NET", "MVC", "Entity Framework", "LINQ", "SQL Server", "Web APIs",
    "Visual Studio", "Azure DevOps", "Ethereum", "Solidity", "Smart Contracts", "Web3.js", "Hyperledger",
    "Cryptography", "DApps", "Consensus Algorithms", "Manual Testing", "Test Cases", "Bug Tracking", "SDLC",
    "STLC", "Regression Testing", "Performance Testing", "QA Processes"
]

job_categories = [
    'Data Science',
    'HR',
    'Advocate',
    'Arts',
    'Web Designing',
    'Mechanical Engineer',
    'Sales',
    'Health and fitness',
    'Civil Engineer',
    'Java Developer',
    'Business Analyst',
    'SAP Developer',
    'Automation Testing',
    'Electrical Engineering',
    'Operations Manager',
    'Python Developer',
    'DevOps Engineer',
    'Network Security Engineer',
    'PMO',
    'Database',
    'Hadoop',
    'ETL Developer',
    'DotNet Developer',
    'Blockchain',
    'Testing'
]

try:
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df['Resume'] = df['Resume'].astype(str)
    job_skill_map = df.groupby('Category')['Resume'].apply(lambda x: ' '.join(x)).reset_index()
    vectorized_resume = vectorizer.transform(job_skill_map['Resume'])
    feature_names = vectorizer.get_feature_names_out()
except FileNotFoundError:
    exit(1)
except:
    exit(1)

def get_top_skills(vector, top_n=10):
    top_indices = np.array(vector.toarray())[0].argsort()[-top_n:][::-1]
    all_top_skills = [feature_names[i] for i in top_indices]
    filtered_skills = [skill for skill in all_top_skills if any(skill.lower() == s.lower() for s in SKILLS)]
    final_skills = [next(s for s in SKILLS if s.lower() == skill.lower()) for skill in filtered_skills]
    return final_skills[:top_n]

category_skills_dict = {row['Category']: get_top_skills(vectorized_resume[i]) for i, row in job_skill_map.iterrows()}
category_skills_dict = {k: v for k, v in category_skills_dict.items() if v}

def load_users_from_csv(file_path=USERS_CSV):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['username', 'password', 'role'])
            writer.writerow(['admin@gmail.com', 'admin123', 'admin'])
    users = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            users[row['username']] = {'password': row['password'], 'role': row['role']}
    return users

def add_user(email, password, role='user'):
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email format"
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    users = load_users_from_csv()
    if email in users:
        return False, "Email already registered"
    with open(USERS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([email, password, role])
    return True, "User registered successfully"

def cleaned_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    return text

def extract_skills(text):
    text_words = set(re.findall(r'\b[\w\s&]+(?=\b|$)', text.lower()))
    return [skill for skill in SKILLS if skill.lower() in text_words]

def save_resume_data(email, filename, skills, category, upload_date):
    new_entry = {"email": email, "filename": filename, "skills": skills, "category": category, "upload_date": upload_date}
    data = []
    if os.path.exists(PARSED_RESUMES_JSON):
        with open(PARSED_RESUMES_JSON, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
    data.append(new_entry)
    with open(PARSED_RESUMES_JSON, 'w') as f:
        json.dump(data, f, indent=4)

def process_resume_upload(file):
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        with open(filepath, 'rb') as f:
            text = extract_text_from_pdf(f)
        cleaned = cleaned_text(text)
        vect_text = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vect_text)[0]
        skills = extract_skills(text)
        upload_date = datetime.utcnow().isoformat()
        return prediction, skills, filename, upload_date
    return None, [], None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', '').strip()
        if not all([username, password, role]):
            flash("All fields are required!", 'error')
            return render_template('login.html')
        users = load_users_from_csv()
        user = users.get(username)
        if user is None:
            flash("User not found!", 'error')
            return render_template('login.html')
        if user['password'] == password and user['role'] == role:
            session.permanent = True
            session['username'] = username
            session['role'] = role
            flash("Login successful!", 'success')
            return redirect('/admin' if role == 'admin' else '/user')
        else:
            flash("Invalid credentials or role!", 'error')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        success, message = add_user(email, password, role)
        if success:
            flash(message, 'success')
            session['username'] = email
            session['role'] = role
            return redirect('/user' if role == 'user' else '/admin')
        else:
            flash(message, 'error')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!", 'success')
    return redirect('/')

@app.route('/user', methods=['GET', 'POST'])
def user_dashboard():
    if session.get('role') != 'user':
        flash("Access denied!", 'error')
        return redirect('/')
    prediction = None
    skills = []
    user_resumes = []
    with open(PARSED_RESUMES_JSON, 'r') as f:
        all_resumes = json.load(f)
        if not isinstance(all_resumes, list):
            all_resumes = []
        user_resumes = [r for r in all_resumes if r.get('email') == session['username']]
    if request.method == 'POST':
        uploaded_file = request.files.get('resume_file')
        prediction, skills, filename, upload_date = process_resume_upload(uploaded_file)
        if filename:
            save_resume_data(session['username'], filename, skills, prediction, upload_date)
            flash("Resume uploaded successfully!", 'success')
            return redirect('/user')
        else:
            flash("Invalid file format. Please upload a PDF file.", 'error')
    return render_template('dashboard_user.html', username=session['username'], prediction=prediction, skills=skills, user_resumes=user_resumes)

@app.route('/upload', methods=['GET', 'POST'])
def upload_resume():
    if session.get('role') != 'user':
        flash("Access denied!", 'error')
        return redirect('/')
    if request.method == 'POST':
        uploaded_file = request.files.get('resume_file')
        prediction, skills, filename, upload_date = process_resume_upload(uploaded_file)
        if filename:
            save_resume_data(session['username'], filename, skills, prediction, upload_date)
            flash("Resume uploaded successfully!", 'success')
            return redirect('/user')
        else:
            flash("Invalid file format. Please upload a PDF file.", 'error')
            return redirect('/upload')
    return render_template('upload_resume.html')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df['Resume'] = df['Resume'].astype(str)
    job_skill_map = df.groupby('Category')['Resume'].apply(lambda x: ' '.join(x)).reset_index()
    vectorized_resume = vectorizer.transform(job_skill_map['Resume'])
    feature_names = vectorizer.get_feature_names_out()
except FileNotFoundError:
    exit(1)
except:
    exit(1)

def get_top_skills(vector, top_n=10):
    top_indices = np.array(vector.toarray())[0].argsort()[-top_n:][::-1]
    all_top_skills = [feature_names[i] for i in top_indices]
    filtered_skills = [skill for skill in all_top_skills if any(skill.lower() == s.lower() for s in SKILLS)]
    final_skills = [next(s for s in SKILLS if s.lower() == skill.lower()) for skill in filtered_skills]
    return final_skills[:top_n]

category_skills_dict = {row['Category']: get_top_skills(vectorized_resume[i]) for i, row in job_skill_map.iterrows()}
category_skills_dict = {k: v for k, v in category_skills_dict.items() if v}

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if session.get('role') != 'user':
        flash("Access denied! Please log in as a user.", 'error')
        return redirect('/')
    recommended_skills = []
    selected_category = None
    if request.method == 'POST':
        selected_category = request.form.get('category', '').strip()
        if not selected_category:
            flash("Please select a job category.", 'error')
        elif selected_category not in category_skills_dict:
            flash(f"Invalid job category: {selected_category}.", 'error')
        else:
            recommended_skills = category_skills_dict.get(selected_category, [])
            if not recommended_skills:
                flash(f"No relevant skills found for category '{selected_category}'. Try another category.", 'info')
            else:
                flash(f"Recommended skills for {selected_category} loaded successfully!", 'success')
    job_categories = sorted(category_skills_dict.keys())
    if not job_categories:
        flash("No job categories available. Please contact the administrator.", 'error')
    return render_template('recommendations.html', job_categories=job_categories, recommended_skills=recommended_skills, selected_category=selected_category)

@app.route('/admin')
def admin_dashboard():
    if session.get('role') != 'admin':
        flash("Access denied!", 'error')
        return redirect('/')
    resumes = []
    with open(PARSED_RESUMES_JSON, 'r') as f:
        resumes = json.load(f)
        if not isinstance(resumes, list):
            resumes = []
    users = load_users_from_csv()
    total_resumes = len(resumes)
    total_users = len(users)
    recent_uploads = len([r for r in resumes if datetime.fromisoformat(r.get('upload_date', '1970-01-01T00:00:00')) > datetime.utcnow() - timedelta(days=7)])
    category_counts = {}
    for resume in resumes:
        category = resume.get('category', 'Unknown')
        category_counts[category] = category_counts.get(category, 0) + 1
    chart_labels = json.dumps(list(category_counts.keys()))
    chart_data = json.dumps(list(category_counts.values()))
    return render_template('dashboard_admin.html', admin_email=session['username'], total_resumes=total_resumes, total_users=total_users, recent_uploads=recent_uploads, chart_labels=chart_labels, chart_data=chart_data)

from urllib.parse import quote
@app.route('/resumes')
def view_resumes():
    resumes_folder = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    try:
        resume_files = os.listdir(resumes_folder)
    except:
        resume_files = []
    resumes = []
    for filename in resume_files:
        if filename.lower().endswith('.pdf'): 
            resumes.append({
                'filename': filename,
                'path': url_for('static', filename=f'resumes/{filename}'), 
            })
    return render_template('view_resumes.html', resumes=resumes)

from urllib.parse import unquote
@app.route('/download_resume/<path:filename>')
def download_resume(filename):
    decoded_filename = unquote(filename)
    file_path = os.path.join('static', 'resumes', decoded_filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('File not found.', 'error')
        return redirect(url_for('view_resumes'))

@app.route('/manage_users', methods=['GET', 'POST'])
def manage_users():
    if session.get('role') != 'admin':
        flash("Access denied!", 'error')
        return redirect('/')
    users = load_users_from_csv()
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        if action == 'delete' and username in users:
            with open(USERS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['username', 'password', 'role'])
                for u, data in users.items():
                    if u != username:
                        writer.writerow([u, data['password'], data['role']])
            flash(f"User {username} deleted successfully!", 'success')
        else:
            flash("Invalid action or user not found!", 'error')
        return redirect('/manage_users')
    return render_template('manage_users.html', users=users)

@app.route('/filter_by_skills', methods=['GET', 'POST'])
def filter_by_skills():
    if session.get('role') != 'admin':
        flash("Access denied! Please log in as admin.", 'error')
        return redirect('/login')
    selected_category = request.form.get('category') or request.args.get('category')
    matched_resumes = []
    resumes = []
    with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content:
            resumes = json.loads(content)
    if selected_category:
        for resume in resumes:
            if resume.get('category') == selected_category:
                matched_resumes.append(resume)
    else:
        flash("No category selected!", 'error')
    return render_template('filter_by_skills.html', job_categories=job_categories, matched_resumes=matched_resumes, selected_category=selected_category)

@app.route('/export_filtered_resumes')
def export_filtered_resumes():
    if session.get('role') != 'admin':
        flash("Access denied!", 'error')
        return redirect('/')
    category = request.args.get('category')
    if not category:
        flash("No category selected for export!", 'error')
        return redirect('/filter_by_skills')
    resumes = []
    with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            resumes = []
        else:
            resumes = json.load(f)
            if not isinstance(resumes, list):
                resumes = []
    matched = [r for r in resumes if set(category_skills_dict.get(category, [])).issubset(set(map(str.lower, r.get('skills', []))))]
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Email', 'Filename', 'Skills'])
    for r in matched:
        writer.writerow([r.get('email', ''), r.get('filename', ''), ', '.join(r.get('skills', []))])
    output.seek(0)
    return send_file(BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=f'{category}_resumes.csv')

if __name__ == '__main__':
    app.run(debug=True)