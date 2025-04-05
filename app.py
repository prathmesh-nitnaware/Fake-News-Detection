from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import sys

# Add the current directory to path to find the models module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Make sure models directory exists first
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created models directory at {models_dir}")
    
    # Create an empty __init__.py file to make it a proper Python package
    with open(os.path.join(models_dir, '__init__.py'), 'w') as f:
        f.write('# This file makes the models directory a Python package')

# Import from local models directory
try:
    from models.detector import FakeNewsDetector
except ImportError as e:
    print(f"ImportError: {e}")
    print("Make sure detector.py is in the models directory.")
    # Create a placeholder detector.py if it doesn't exist
    detector_path = os.path.join(models_dir, 'detector.py')
    if not os.path.exists(detector_path):
        print(f"detector.py not found. Creating placeholder at {detector_path}")
        with open(detector_path, 'w') as f:
            # Copy the detector.py code here
            pass  # You would need to put detector.py code here if needed

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 116 * 1024 * 1024  # 16MB max

# Initialize detector
detector = FakeNewsDetector()

# Check if models directory exists and load models
models_path = './fake_news_models/'
if os.path.exists(models_path):
    try:
        models_loaded = detector.load_models()
        if models_loaded:
            print("Models loaded successfully")
        else:
            print("Failed to load models. System will use heuristics.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("System will use heuristics.")
else:
    print("Models directory not found. System will use heuristics.")
    # Create the directory for future model saving
    os.makedirs(models_path, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    if request.method == 'POST':
        # Debug info - helps identify what's being received
        print("Form data received:", request.form)
        print("Content-Type:", request.headers.get('Content-Type'))
        
        # Try different ways to get the data based on content type
        if request.is_json:
            data = request.get_json()
            news_content = data.get('news_content', '')
            news_url = data.get('news_url', '')
        else:
            # For form data and other content types
            news_content = request.form.get('news_content', '')
            news_url = request.form.get('news_url', '')
        
        print(f"Parsed data: content='{news_content[:50]}...' (length: {len(news_content)}), url='{news_url}'")
        
        # Validate inputs
        if not news_content and not news_url:
            return jsonify({
                'error': 'Please provide either text content or a URL'
            }), 400
            
        # Process input
        try:
            if news_url:
                result = detector.predict(url=news_url)
            else:
                result = detector.predict(input_text=news_content)
                
            if 'error' in result:
                return jsonify({'error': result['error']}), 400
                
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"Error processing request: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f"Processing error: {str(e)}"}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Admin endpoint to train the model with new data"""
    if request.method == 'POST':
        if 'dataset' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['dataset']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            
            # Ensure upload directory exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
                
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Simple validation - check if it's a CSV
            if not filename.endswith('.csv'):
                return jsonify({'error': 'Please upload a CSV file'}), 400
                
            try:
                import pandas as pd
                data = pd.read_csv(filepath)
                
                # Check if required columns exist
                if 'text' not in data.columns or 'label' not in data.columns:
                    return jsonify({
                        'error': 'Dataset must contain "text" and "label" columns'
                    }), 400

                print(f"Training started on  samples...")
                
                # Train the model
                training_results = detector.train(data['text'], data['label'])
                
                print(f"Training done....")

                # Save the trained models
                detector.save_models()

                print(f"done saving to model......")
                
                return jsonify({
                    'success': True,
                    'message': 'Model trained successfully',
                    'results': training_results
                })
                
            except Exception as e:
                import traceback
                print(f"Training error: {str(e)}")
                print(traceback.format_exc())
                return jsonify({'error': f'Training failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create directories if they don't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"Created templates directory at {templates_dir}")
        # Create a minimal index.html if it doesn't exist
        index_path = os.path.join(templates_dir, 'index.html')
        if not os.path.exists(index_path):
            with open(index_path, 'w') as f:
                f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
</head>
<body>
    <h1>Fake News Detector</h1>
    <form id="text-form">
        <h2>Analyze Text</h2>
        <textarea id="news_content" rows="8" cols="80"></textarea>
        <button type="submit">Analyze</button>
    </form>
    <form id="url-form">
        <h2>Analyze URL</h2>
        <input type="url" id="news_url" placeholder="https://example.com/news-article">
        <button type="submit">Analyze</button>
    </form>
    <div id="results"></div>
    
    <script>
        document.getElementById('text-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const text = document.getElementById('news_content').value;
            fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: new URLSearchParams({news_content: text})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('results').innerHTML = 
                    `<h3>Score: ${data.credibility_score}</h3>
                     <h3>Category: ${data.category}</h3>
                     <pre>${JSON.stringify(data, null, 2)}</pre>`;
            });
        });
        
        document.getElementById('url-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const url = document.getElementById('news_url').value;
            fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: new URLSearchParams({news_url: url})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('results').innerHTML = 
                    `<h3>Score: ${data.credibility_score}</h3>
                     <h3>Category: ${data.category}</h3>
                     <pre>${JSON.stringify(data, null, 2)}</pre>`;
            });
        });
    </script>
</body>
</html>''')
                print(f"Created a basic index.html file")
        
    app.run(debug=True)