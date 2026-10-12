import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained KNeighborsRegressor model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "knnRe.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --accent: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 760px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 2.5rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
        }

        header {
            margin-bottom: 2rem;
            text-align: center;
        }

        header h1 {
            font-size: 1.875rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        header p {
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-size: 0.95rem;
        }

        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            background-color: #0f172a;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.25);
        }

        .full-width {
            grid-column: 1 / -1;
        }

        button {
            padding: 0.85rem 1.5rem;
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            font-size: 1rem;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.05s ease;
            margin-top: 0.5rem;
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        button:active {
            transform: scale(0.99);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 0.5rem;
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--accent);
            text-align: center;
        }

        .result-box h2 {
            font-size: 1.1rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .result-box span {
            display: block;
            margin-top: 0.5rem;
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--accent);
        }

        .error-box {
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
            color: #fca5a5;
            text-align: center;
            font-size: 0.95rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>KNN Regression Predictor</h1>
            <p>Input student attributes to calculate the predicted score</p>
        </header>

        <form action="/" method="POST">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" step="any" id="age" name="age" required value="{{ request.form.get('age', '') }}" placeholder="e.g. 18">
            </div>

            <div class="form-group">
                <label for="gender">Gender (Encoded)</label>
                <input type="number" step="any" id="gender" name="gender" required value="{{ request.form.get('gender', '') }}" placeholder="0 or 1">
            </div>

            <div class="form-group">
                <label for="course">Course (Encoded)</label>
                <input type="number" step="any" id="course" name="course" required value="{{ request.form.get('course', '') }}" placeholder="Numerical Code">
            </div>

            <div class="form-group">
                <label for="study_hours">Study Hours / Week</label>
                <input type="number" step="any" id="study_hours" name="study_hours" required value="{{ request.form.get('study_hours', '') }}" placeholder="e.g. 15.5">
            </div>

            <div class="form-group">
                <label for="class_attendance">Class Attendance (%)</label>
                <input type="number" step="any" id="class_attendance" name="class_attendance" required value="{{ request.form.get('class_attendance', '') }}" placeholder="0 - 100">
            </div>

            <div class="form-group">
                <label for="internet_access">Internet Access</label>
                <input type="number" step="any" id="internet_access" name="internet_access" required value="{{ request.form.get('internet_access', '') }}" placeholder="0 or 1">
            </div>

            <div class="form-group">
                <label for="sleep_hours">Sleep Hours</label>
                <input type="number" step="any" id="sleep_hours" name="sleep_hours" required value="{{ request.form.get('sleep_hours', '') }}" placeholder="e.g. 7.5">
            </div>

            <div class="form-group">
                <label for="sleep_quality">Sleep Quality Rating</label>
                <input type="number" step="any" id="sleep_quality" name="sleep_quality" required value="{{ request.form.get('sleep_quality', '') }}" placeholder="Scale rating">
            </div>

            <div class="form-group">
                <label for="study_method">Study Method (Encoded)</label>
                <input type="number" step="any" id="study_method" name="study_method" required value="{{ request.form.get('study_method', '') }}" placeholder="Numerical Code">
            </div>

            <div class="form-group">
                <label for="facility_rating">Facility Rating</label>
                <input type="number" step="any" id="facility_rating" name="facility_rating" required value="{{ request.form.get('facility_rating', '') }}" placeholder="Scale rating">
            </div>

            <div class="form-group full-width">
                <label for="exam_difficulty">Exam Difficulty (Encoded)</label>
                <input type="number" step="any" id="exam_difficulty" name="exam_difficulty" required value="{{ request.form.get('exam_difficulty', '') }}" placeholder="Numerical Code">
            </div>

            <button type="submit" class="full-width">Generate Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <h2>Estimated Target Value</h2>
            <span>{{ "%.2f"|format(prediction) }}</span>
        </div>
        {% elif error %}
        <div class="error-box">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

FEATURE_ORDER = [
    "age",
    "gender",
    "course",
    "study_hours",
    "class_attendance",
    "internet_access",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            features = [float(request.form[field]) for field in FEATURE_ORDER]
            features_array = np.array([features])
            
            # Run model prediction
            raw_prediction = model.predict(features_array)
            prediction = float(raw_prediction[0])
        except KeyError as e:
            error = f"Missing required parameter: {str(e)}"
        except ValueError:
            error = "Invalid format: All features must be numeric values."
        except Exception as e:
            error = f"Inference error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
