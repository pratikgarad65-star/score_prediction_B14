import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the KNN Regressor model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "knnRe.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Features expected by knnRe.pkl in strict order
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
    "exam_difficulty",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KNN Performance Predictor</title>
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #151c2e;
            --border: #232d45;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --success: #10b981;
            --error: #ef4444;
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
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 1rem;
            width: 100%;
            max-width: 720px;
            padding: 2.5rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
        }
        header {
            text-align: center;
            margin-bottom: 2rem;
        }
        header h1 {
            font-size: 1.75rem;
            font-weight: 700;
        }
        header p {
            color: var(--text-sub);
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }
        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.25rem;
        }
        .field {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        input {
            width: 100%;
            padding: 0.75rem 1rem;
            background: #0d1220;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: border-color 0.15s ease;
        }
        input:focus {
            outline: none;
            border-color: var(--primary);
        }
        button {
            padding: 0.85rem 1.5rem;
            background: var(--primary);
            color: white;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: background 0.15s ease;
            margin-top: 0.5rem;
        }
        button:hover {
            background: var(--primary-hover);
        }
        .result {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
            text-align: center;
        }
        .result span {
            display: block;
            margin-top: 0.25rem;
            font-size: 2rem;
            font-weight: 700;
            color: var(--success);
        }
        .error {
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error);
            color: #fca5a5;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="card">
        <header>
            <h1>Student Score Predictor</h1>
            <p>KNN Regression inference model</p>
        </header>

        <form action="/" method="POST">
            <div class="field">
                <label for="age">Age</label>
                <input type="number" step="any" id="age" name="age" required value="{{ request.form.get('age', '') }}" placeholder="e.g. 19">
            </div>
            <div class="field">
                <label for="gender">Gender (Numeric)</label>
                <input type="number" step="any" id="gender" name="gender" required value="{{ request.form.get('gender', '') }}" placeholder="0 or 1">
            </div>
            <div class="field">
                <label for="course">Course (Numeric)</label>
                <input type="number" step="any" id="course" name="course" required value="{{ request.form.get('course', '') }}" placeholder="e.g. 1">
            </div>
            <div class="field">
                <label for="study_hours">Study Hours / Week</label>
                <input type="number" step="any" id="study_hours" name="study_hours" required value="{{ request.form.get('study_hours', '') }}" placeholder="e.g. 15.0">
            </div>
            <div class="field">
                <label for="class_attendance">Class Attendance (%)</label>
                <input type="number" step="any" id="class_attendance" name="class_attendance" required value="{{ request.form.get('class_attendance', '') }}" placeholder="0 - 100">
            </div>
            <div class="field">
                <label for="internet_access">Internet Access</label>
                <input type="number" step="any" id="internet_access" name="internet_access" required value="{{ request.form.get('internet_access', '') }}" placeholder="0 or 1">
            </div>
            <div class="field">
                <label for="sleep_hours">Sleep Hours</label>
                <input type="number" step="any" id="sleep_hours" name="sleep_hours" required value="{{ request.form.get('sleep_hours', '') }}" placeholder="e.g. 7.5">
            </div>
            <div class="field">
                <label for="sleep_quality">Sleep Quality Rating</label>
                <input type="number" step="any" id="sleep_quality" name="sleep_quality" required value="{{ request.form.get('sleep_quality', '') }}" placeholder="Scale rating">
            </div>
            <div class="field">
                <label for="study_method">Study Method (Numeric)</label>
                <input type="number" step="any" id="study_method" name="study_method" required value="{{ request.form.get('study_method', '') }}" placeholder="e.g. 1">
            </div>
            <div class="field">
                <label for="facility_rating">Facility Rating</label>
                <input type="number" step="any" id="facility_rating" name="facility_rating" required value="{{ request.form.get('facility_rating', '') }}" placeholder="Scale rating">
            </div>
            <div class="field full-width">
                <label for="exam_difficulty">Exam Difficulty (Numeric)</label>
                <input type="number" step="any" id="exam_difficulty" name="exam_difficulty" required value="{{ request.form.get('exam_difficulty', '') }}" placeholder="Scale rating">
            </div>

            <button type="submit" class="full-width">Calculate Score</button>
        </form>

        {% if prediction is not none %}
        <div class="result">
            <div>Predicted Outcome</div>
            <span>{{ "%.2f"|format(prediction) }}</span>
        </div>
        {% elif error %}
        <div class="error">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            features = [float(request.form[col]) for col in FEATURE_ORDER]
            features_array = np.array([features], dtype=np.float64)
            prediction = float(model.predict(features_array)[0])
        except KeyError as e:
            error = f"Missing feature value: {str(e)}"
        except ValueError:
            error = "Invalid format: All entries must be numeric."
        except Exception as e:
            error = f"Inference failure: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
