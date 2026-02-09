import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class CareerGuidanceSystem:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'math_score', 'science_score', 'logical_reasoning',
            'coding_interest', 'spatial_ability', 'communication_score',
            'leadership_score', 'resume_skills_score'
        ]

        self.career_mapping = {
            'Software Engineer': 0, 'Data Scientist': 1, 'Mechanical Engineer': 2,
            'Civil Engineer': 3, 'Entrepreneur/Business': 4, 'Electronics Engineer': 5,
            'AI/ML Engineer': 6, 'Project Manager': 7, 'Sales/Marketing': 8, 'Consultant': 9
        }
        self.reverse_career_mapping = {v: k for k, v in self.career_mapping.items()}

    def load_real_training_data(self):
        csv_files = ['students.csv', 'engineering_student_journey.csv']
        for csv_path in csv_files:
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    print(f"✅ Loaded YOUR {len(df)} records")
                    return self._ensure_features(df)
                except:
                    pass
        return self._create_dataset()

    def _create_dataset(self):
        np.random.seed(42)
        n = 10000

        data = {}
        for feature in self.feature_names:
            data[feature] = np.clip(np.random.normal(70, 15, n), 0, 100)

        careers = np.random.choice(list(self.career_mapping.keys()), n)
        data['career'] = pd.Series(careers).map(self.career_mapping).fillna(0).astype(int)
        df = pd.DataFrame(data)
        df.to_csv('dataset.csv', index=False)
        print(f"✅ Created 10K dataset")
        return df

    def _ensure_features(self, df):
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 50.0
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(50.0)
        if 'career' not in df.columns:
            df['career'] = 0
        return df[self.feature_names + ['career']]

    def train_model(self, df):
        X = df[self.feature_names].values.astype(np.float64)
        y = df['career'].values.astype(int)

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 🔥 PRODUCTION MODEL - SHARP PREDICTIONS
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        test_acc = accuracy_score(y_test, self.model.predict(X_test))

        print(f"🚀 Model trained! Train: {train_acc:.1%} | Test: {test_acc:.1%}")
        self.save_model()
        return test_acc

    def save_model(self):
        joblib.dump({
            'model': self.model, 'scaler': self.scaler,
            'feature_names': self.feature_names,
            'career_mapping': self.career_mapping
        }, 'career_model.joblib')

    def load_model(self):
        if os.path.exists('career_model.joblib'):
            data = joblib.load('career_model.joblib')
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data.get('feature_names', self.feature_names)
            self.career_mapping = data.get('career_mapping', self.career_mapping)
            self.reverse_career_mapping = {v: k for k, v in self.career_mapping.items()}
            print("✅ Model loaded!")
            return True
        return False

    def predict_career(self, profile):
        features = [profile.get(f, 50.0) for f in self.feature_names]
        X_pred = np.array([features]).astype(np.float64)

        probs = self.model.predict_proba(self.scaler.transform(X_pred))[0]
        top3 = np.argsort(probs)[-3:][::-1]

        recommendations = []
        for i, idx in enumerate(top3):
            recommendations.append({
                'rank': i + 1,
                'career': self.reverse_career_mapping[int(idx)],
                'confidence': f"{probs[idx] * 100:.1f}%"
            })

        return {
            'top_recommendation': recommendations[0]['career'],
            'all_recommendations': recommendations
        }
