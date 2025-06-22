from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import sys
import traceback

# ---------------------- Setup Directories -----------------------

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure models directory exists
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

# Ensure __init__.py exists to make it a package
init_file = os.path.join(models_dir, '__init__.py')
if not os.path.exists(init_file):
    with open(init_file, 'w') as f:
        f.write('# makes models a package\n')

# Ensure fake_news_models and uploads directories exist
os.makedirs('fake_news_models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# ---------------------- Import Detector -------------------------

try:
    from models.detector import FakeNewsDetector
except ImportError as e:
    print(f"ImportError: {e}")
    detector_path = os.path.join(models_dir, 'detector.py')
    if not os.path.exists(detector_path):
        with open(detector_path, 'w') as f:
            f.write("# Placeholder for FakeNewsDetector class.\n")
    print("Please add 'detector.py' with the FakeNewsDetector class in models/ directory.")
    FakeNewsDetector = None

# ------------------------ Flask App Init ------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ------------------------ Load Detector -------------------------

detector = FakeNewsDetector() if FakeNewsDetector else None

if detector:
    try:
        models_loaded = detector.load_models()
        print("Models loaded successfully." if models_loaded else "Failed to load models.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Using fallback or heuristics.")
else:
    print("Detector not available. Please define FakeNewsDetector in detector.py.")

# ------------------------ Routes -------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    if not detector:
        return jsonify({'error': 'Model not available'}), 500

    if request.method == 'POST':
        print("Form data received:", request.form)

        if request.is_json:
            data = request.get_json()
            news_content = data.get('news_content', '')
            news_url = data.get('news_url', '')
        else:
            news_content = request.form.get('news_content', '')
            news_url = request.form.get('news_url', '')

        if not news_content and not news_url:
            return jsonify({'error': 'Please provide either text content or a URL'}), 400

        try:
            if news_url:
                result = detector.predict(url=news_url)
            else:
                result = detector.predict(input_text=news_content)

            if 'error' in result:
                return jsonify({'error': result['error']}), 400

            return jsonify(result)

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f"Processing error: {str(e)}"}), 500

@app.route('/train', methods=['POST'])
def train_model():
    if not detector:
        return jsonify({'error': 'Model not available'}), 500

    if 'dataset' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if not filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400

    try:
        import pandas as pd
        data = pd.read_csv(filepath)

        if 'text' not in data.columns or 'label' not in data.columns:
            return jsonify({
                'error': 'Dataset must contain "text" and "label" columns'
            }), 400

        training_results = detector.train(data['text'], data['label'])
        detector.save_models()

        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'results': training_results
        })

    except Exception as e:
        print(f"Training error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

# ------------------------ Bootstrap Index if Needed ------------------------

index_path = os.path.join('templates', 'index.html')
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
        print("Created a basic index.html file")

# ------------------------ Run the App ------------------------

if __name__ == '__main__':
    app.run(debug=True)
