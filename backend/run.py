from career_guidance_system import CareerGuidanceSystem

if __name__ == "__main__":
    print(" AI Career Guidance System Starting...")
    system = CareerGuidanceSystem(model_type='knn')
    df = system.load_training_data()
    system.train_model(df)

    user = {
        'math_score': 88, 'communication_score': 85, 'problem_solving': 92,
        'coding_experience': 2.5, 'creativity_score': 72,
        'leadership_score': 80, 'years_experience': 1
    }

    results = system.predict_career(user)
    print(f"\n TOP RECOMMENDATION: {results['top_recommendation']}")
