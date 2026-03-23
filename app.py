from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os

sys.path.append('backend')

try:
    from career_guidance_system import CareerGuidanceSystem

    career_system = CareerGuidanceSystem()
    if not career_system.load_model():
        print("Training model...")
        df = career_system.load_real_training_data()
        career_system.train_model(df)
    ML_READY = True
except:
    ML_READY = False
    print("⚠️ Using demo data")

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/career-test')
def career_test():
    return render_template('career-test.html')


@app.route('/pricing')
def pricing():
    return render_template('pricing.html')


@app.route('/colleges')
def colleges():
    return render_template('colleges.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_career():
    try:
        data = request.json or {}

        # Your ML model's exact 8 features
        profile = {
            'math_score': float(data.get('math_score', 75)),
            'science_score': float(data.get('science_score', 70)),
            'logical_reasoning': float(data.get('logical_reasoning', 80)),
            'coding_interest': float(data.get('coding_interest', 85)),
            'spatial_ability': float(data.get('spatial_ability', 75)),
            'communication_score': float(data.get('communication_score', 65)),
            'leadership_score': float(data.get('leadership_score', 90)),
            'resume_skills_score': float(data.get('resume_skills_score', 80))
        }

        if ML_READY:
            result = career_system.predict_career(profile)
        else:
            # Demo results
            result = {
                'all_recommendations': [
                    {'rank': 1, 'career': 'Software Engineer', 'confidence': '92.3%'},
                    {'rank': 2, 'career': 'AI/ML Engineer', 'confidence': '87.1%'},
                    {'rank': 3, 'career': 'Data Scientist', 'confidence': '84.5%'}
                ]
            }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 AI Career Pro LIVE!")
    print("ML Status:", "✅ READY" if ML_READY else "❌ Demo mode")
    app.run(debug=True, port=5001)
