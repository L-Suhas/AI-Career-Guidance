from career_guidance_system import CareerGuidanceSystem


def validate_numeric_input(prompt, min_val=0, max_val=100, default=50):
    """Validate numeric input (0-100) with error messages"""
    while True:
        try:
            value = input(prompt).strip()
            if value == '' or value.lower() == 'default':
                return default

            num_value = float(value)
            if not (min_val <= num_value <= max_val):
                print(f" ERROR: Value must be between {min_val} and {max_val}")
                print(f" Enter a number like: {default}")
                continue
            return num_value

        except ValueError:
            print(" ERROR: Please enter a valid NUMBER (e.g., 75, 85.5)")
            print(" Press Enter for default or type a number 0-100")


def validate_yes_no_input(prompt, default='n'):
    """ Validate y/n input"""
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        elif response == '':
            return default == 'y'
        else:
            print(" ERROR: Please enter 'y' (yes) or 'n' (no)")
            print(" Press Enter for default")


def get_validated_profile():
    """✅ Get fully validated user profile"""
    print("\n" + "=" * 60)
    print("🎓 CAREER ASSESSMENT - ENTER YOUR PROFILE")
    print("(All values 0-100, Press Enter for defaults)")
    print("=" * 60)

    profile = {}

    # Numeric inputs with validation
    profile['math_score'] = validate_numeric_input("📚 Math % (0-100) [75]: ", 0, 100, 75)
    profile['science_score'] = validate_numeric_input("🔬 Science % (0-100) [70]: ", 0, 100, 70)
    profile['logical_reasoning'] = validate_numeric_input("🧠 Logic (0-100) [65]: ", 0, 100, 65)
    profile['coding_interest'] = validate_numeric_input("💻 Coding interest (0-100) [50]: ", 0, 100, 50)
    profile['communication_score'] = validate_numeric_input("🗣️ Communication (0-100) [60]: ", 0, 100, 60)
    profile['resume_skills_score'] = validate_numeric_input("📄 Skills score (0-100) [50]: ", 0, 100, 50)

    # Yes/No inputs
    profile['spatial_ability'] = 85 if validate_yes_no_input("🏗️ Like design/building? (y/n) [n]: ") else 40
    profile['leadership_score'] = 85 if validate_yes_no_input("👥 Led teams/business? (y/n) [n]: ") else 50

    print("Profile validated successfully!")
    return profile


def main():
    print("🚀 Starting Career Guidance System...")
    print("📊 Loading & training model...")

    try:
        system = CareerGuidanceSystem(model_type='rf')

        # Load or train model
        if not system.load_model():
            df = system.load_real_training_data()
            system.train_model(df)

        print("\n🎓 VALIDATED INTERACTIVE ASSESSMENT STARTED")
        print("=" * 60)

        while True:
            # ✅ VALIDATED INPUT COLLECTION
            profile = get_validated_profile()

            # Get predictions
            results = system.predict_career(profile)

            print("\n" + "=" * 60)
            print("🎯 YOUR CAREER RECOMMENDATIONS:")
            print(f"🏆 #1: {results['top_recommendation']:<25} {results['all_recommendations'][0]['confidence']}")
            print(
                f"🥈 #2: {results['all_recommendations'][1]['career']:<25} {results['all_recommendations'][1]['confidence']}")
            print(
                f"🥉 #3: {results['all_recommendations'][2]['career']:<25} {results['all_recommendations'][2]['confidence']}")

            print("\n💡 Pro Tip:")
            print("   High spatial_ability + leadership_score → Entrepreneur/Business")
            print("   High coding_interest + math_score → Software Engineer")

            if input("\n🔄 Another prediction? (y/n): ").lower() not in ['y', 'yes']:
                print("\n👋 Thanks for using AI Career Guidance System!")
                break

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        print("💡 Make sure you have: pip install pandas scikit-learn numpy joblib")


if __name__ == "__main__":
    main()
