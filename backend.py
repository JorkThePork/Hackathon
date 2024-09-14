from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import Dict, Any

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit-data', methods=['POST'])
def submit_data() -> Dict[str, Any]:
    try:
        data = {
            'age': request.form.get('age'),
            'weight': request.form.get('weight'),
            'lifestyle': request.form.get('lifestyle'),
            'smoking': request.form.get('smoking'),
            'alcohol': request.form.get('alcohol'),
            'condition': request.form.get('condition'),
            'exercise': request.form.get('exercise'),
            'diet': request.form.get('diet'),
            'sleep': request.form.get('sleep'),
            'family_history': request.form.getlist('family_history[]'),
            'family_history_details': request.form.get('family_history_details'),
            'current_symptoms': request.form.get('current_symptoms'),
            'symptom_severity': request.form.get('symptom_severity'),
            'symptom_duration': request.form.get('symptom_duration')
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/llm-analysis', methods=['POST'])
def llm_analysis() -> Dict[str, Any]:
    try:
        # Get the data from the request
        data = request.json

        # Here you would typically send the data to your LLM and get a response
        # For now, we'll just return a placeholder response
        llm_response = {
            'analysis': 'This is a placeholder for the LLM analysis.',
            'recommendations': [
                'Recommendation 1 based on the provided data.',
                'Recommendation 2 based on the provided data.',
                'Recommendation 3 based on the provided data.'
            ],
            'risk_factors': [
                {'factor': 'Factor 1', 'risk_level': 'High'},
                {'factor': 'Factor 2', 'risk_level': 'Medium'},
                {'factor': 'Factor 3', 'risk_level': 'Low'}
            ]
        }

        # Combine the original data with the LLM response
        response = {
            'input_data': data,
            'llm_analysis': llm_response
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)