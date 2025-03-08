from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
app.config['IMAGE_UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS_IMAGE'] = {'png', 'jpg', 'jpeg', 'gif', 'svg'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(app.config['IMAGE_UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Initialize the CKD predictor
class CKDPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'blood_pressure', 'specific_gravity', 'albumin',
            'blood_glucose', 'blood_urea', 'serum_creatinine',
            'sodium', 'potassium', 'hemoglobin'
        ]
        self.is_trained = False
        self.X_test = None
        self.y_test = None
        
        # Try to train with default dataset on initialization
        try:
            default_dataset = os.path.join(DATA_FOLDER, 'ckd_dataset.csv')
            if os.path.exists(default_dataset):
                self.train_with_file(default_dataset)
        except Exception as e:
            print(f"Error loading default dataset: {e}")

    def train_with_file(self, file_path):
        try:
            # Read and preprocess data
            data = pd.read_csv(file_path)
            
            # Ensure all required features are present
            missing_features = [feat for feat in self.feature_names if feat not in data.columns]
            if missing_features:
                return False, f"Missing required features: {', '.join(missing_features)}"
            
            # Extract features and target
            X = data[self.feature_names]
            y = data['target'] if 'target' in data.columns else data['class']
            
            # Split data
            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Calculate accuracy
            X_test_scaled = self.scaler.transform(self.X_test)
            accuracy = self.model.score(X_test_scaled, self.y_test)
            
            return True, accuracy
            
        except Exception as e:
            return False, str(e)

    def predict(self, features):
        if not self.is_trained:
            raise ValueError("Model not trained. Please upload training data first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return prediction[0], probabilities

# Create and train the predictor
predictor = CKDPredictor()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Successfully logged in!', 'success')
            return redirect(url_for('home'))
        flash('Invalid email or password', 'danger')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Train with default dataset if model is not trained
    if not predictor.is_trained:
        try:
            success, accuracy = predictor.train_with_file(os.path.join(DATA_FOLDER, 'ckd_dataset.csv'))
            if success:
                accuracy = round(accuracy * 100, 2)
            else:
                accuracy = None
        except Exception as e:
            accuracy = None
    else:
        # Calculate accuracy on test set
        X_test_scaled = predictor.scaler.transform(predictor.X_test)
        accuracy = round(predictor.model.score(X_test_scaled, predictor.y_test) * 100, 2)

    return render_template('dashboard.html', accuracy=accuracy)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
        
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Validate required fields
        required_fields = [
            'age', 'blood_pressure', 'specific_gravity', 'albumin',
            'blood_glucose', 'blood_urea', 'serum_creatinine',
            'sodium', 'potassium', 'hemoglobin'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            flash(f'Missing required fields: {", ".join(missing_fields)}', 'danger')
            return redirect(url_for('predict'))
        
        # Convert data to numpy array for prediction
        try:
            features = np.array([[
                float(data['age']),
                float(data['blood_pressure']),
                float(data['specific_gravity']),
                float(data['albumin']),
                float(data['blood_glucose']),
                float(data['blood_urea']),
                float(data['serum_creatinine']),
                float(data['sodium']),
                float(data['potassium']),
                float(data['hemoglobin'])
            ]])
        except ValueError:
            flash('Invalid input values. Please ensure all values are numbers.', 'danger')
            return redirect(url_for('predict'))
        
        # Make prediction
        prediction, probabilities = predictor.predict(features)
        probability = float(probabilities[1] if prediction == 1 else probabilities[0])
        
        if request.is_json:
            return jsonify({
                'success': True,
                'prediction': int(prediction),
                'probability': probability
            })
        else:
            return redirect(url_for('result', prediction=prediction, probability=probability))
        
    except Exception as e:
        if request.is_json:
            return jsonify({
                'success': False,
                'error': 'An error occurred during prediction'
            }), 500
        else:
            flash('An error occurred during prediction. Please try again.', 'danger')
            return redirect(url_for('predict'))

@app.route('/result')
@login_required
def result():
    prediction = request.args.get('prediction', type=int)
    probability = request.args.get('probability', type=float)
    if prediction is None or probability is None:
        flash('Invalid prediction parameters', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('result.html', prediction=prediction, probability=probability)

@app.route('/upload_dataset', methods=['POST'])
@login_required
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file'}), 400
            
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Train model with uploaded file
        success, result = predictor.train_with_file(filepath)
        
        if success:
            flash('Dataset uploaded and model trained successfully!', 'success')
            return jsonify({
                'success': True,
                'accuracy': result * 100,
                'redirect': url_for('home')  # Redirect to home page after successful upload
            })
        else:
            # Remove the file if training failed
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'Training failed: {result}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/train_model', methods=['POST'])
@login_required
def train_model():
    try:
        # Get the latest uploaded file
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if not files:
            return jsonify({'error': 'No dataset found'}), 400
        
        latest_file = max([os.path.join(app.config['UPLOAD_FOLDER'], f) for f in files], 
                         key=os.path.getctime)
        
        success, result = predictor.train_with_file(latest_file)
        
        if success:
            return jsonify({'success': True, 'accuracy': round(result * 100, 2)})
        else:
            return jsonify({'error': f'Training failed: {result}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS_IMAGE']):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename))
        return jsonify({'success': True, 'filename': filename}), 200
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
