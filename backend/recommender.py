from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class CareerRecommender:

    def __init__(self, careers_data: list):
        print("Loading SBERT model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("SBERT model loaded successfully!")
        self.careers = careers_data

        print("Computing career embeddings...")
        self.career_texts = [self._career_to_text(c) for c in careers_data]
        self.career_embeddings = self.model.encode(self.career_texts)
        print(f"Ready! {len(careers_data)} career profiles loaded.")

        # RIASEC personality type → career alignment map
        self.riasec_map = {
            "realistic":     ["Cloud Engineer", "Cybersecurity Analyst", "Software Engineer", "Full Stack Developer"],
            "investigative": ["Data Scientist", "Machine Learning Engineer", "Cybersecurity Analyst", "Software Engineer"],
            "artistic":      ["UI/UX Designer", "Digital Marketing Specialist", "Machine Learning Engineer"],
            "social":        ["Product Manager", "Digital Marketing Specialist", "Business Analyst"],
            "enterprising":  ["Product Manager", "Business Analyst", "Digital Marketing Specialist"],
            "conventional":  ["Business Analyst", "Cloud Engineer", "Full Stack Developer"],
        }

        # Keyword → career strong match map
        self.keyword_career_map = {
            # design / creative
            "design": "UI/UX Designer", "figma": "UI/UX Designer", "ui": "UI/UX Designer",
            "ux": "UI/UX Designer", "creative": "UI/UX Designer", "art": "UI/UX Designer",
            "graphic": "UI/UX Designer", "visual": "UI/UX Designer", "wireframe": "UI/UX Designer",
            # data / ml / ai
            "data": "Data Scientist", "machine learning": "Machine Learning Engineer",
            "deep learning": "Machine Learning Engineer", "neural": "Machine Learning Engineer",
            "ai": "Machine Learning Engineer", "nlp": "Machine Learning Engineer",
            "statistics": "Data Scientist", "analytics": "Data Scientist",
            "tensorflow": "Machine Learning Engineer", "pytorch": "Machine Learning Engineer",
            # business / marketing
            "marketing": "Digital Marketing Specialist", "seo": "Digital Marketing Specialist",
            "social media": "Digital Marketing Specialist", "content": "Digital Marketing Specialist",
            "business": "Business Analyst", "management": "Product Manager",
            "strategy": "Product Manager", "product": "Product Manager",
            "leadership": "Product Manager", "mba": "Product Manager",
            # security
            "security": "Cybersecurity Analyst", "hacking": "Cybersecurity Analyst",
            "cyber": "Cybersecurity Analyst", "network": "Cybersecurity Analyst",
            "ethical hacking": "Cybersecurity Analyst",
            # cloud / devops
            "cloud": "Cloud Engineer", "aws": "Cloud Engineer", "azure": "Cloud Engineer",
            "docker": "Cloud Engineer", "devops": "Cloud Engineer", "kubernetes": "Cloud Engineer",
            # web / fullstack
            "web": "Full Stack Developer", "react": "Full Stack Developer",
            "node": "Full Stack Developer", "javascript": "Full Stack Developer",
            "frontend": "Full Stack Developer", "backend": "Full Stack Developer",
            "fullstack": "Full Stack Developer", "website": "Full Stack Developer",
            # software
            "coding": "Software Engineer", "programming": "Software Engineer",
            "software": "Software Engineer", "algorithms": "Software Engineer",
            "development": "Software Engineer",
        }

        # Major / degree → career alignment
        self.major_career_map = {
            "computer science": ["Software Engineer", "Data Scientist", "Machine Learning Engineer", "Full Stack Developer"],
            "information technology": ["Cloud Engineer", "Cybersecurity Analyst", "Full Stack Developer"],
            "software engineering": ["Software Engineer", "Full Stack Developer", "Cloud Engineer"],
            "data science": ["Data Scientist", "Machine Learning Engineer"],
            "artificial intelligence": ["Machine Learning Engineer", "Data Scientist"],
            "design": ["UI/UX Designer"],
            "business": ["Business Analyst", "Product Manager", "Digital Marketing Specialist"],
            "marketing": ["Digital Marketing Specialist", "Product Manager"],
            "economics": ["Business Analyst", "Data Scientist"],
            "mathematics": ["Data Scientist", "Machine Learning Engineer", "Software Engineer"],
            "mba": ["Product Manager", "Business Analyst"],
            "networking": ["Cloud Engineer", "Cybersecurity Analyst"],
            "information security": ["Cybersecurity Analyst"],
        }

    #  Text builders 

    def _career_to_text(self, career: dict) -> str:
        parts = [
            career.get("title", ""),
            career.get("description", ""),
            " ".join(career.get("skills_required", [])),
            " ".join(career.get("related_subjects", [])),
            " ".join(career.get("personality_traits", [])),
            " ".join(career.get("keywords", [])),
            career.get("industry", ""),
        ]
        return " ".join(parts)

    def _profile_to_text(self, profile: dict) -> str:
        parts = [
            f"Student studying {profile.get('degree', '')} in {profile.get('major', '')}",
            f"Interests: {' '.join(profile.get('interests', []))}",
            f"Skills: {' '.join(profile.get('skills', []))}",
            f"Personality: {' '.join(profile.get('personality_traits', []))}",
            f"Goals: {profile.get('goals', '')}",
        ]
        return " ".join(parts)

    #  Scoring components 

    def _keyword_score(self, profile: dict, career: dict) -> float:
        """
        Strong boost when user interests/skills/goals match
        career keywords. This is the main differentiator.
        """
      
        user_text = " ".join([
            " ".join(profile.get("interests", [])),
            " ".join(profile.get("skills", [])),
            profile.get("goals", ""),
            profile.get("major", ""),
        ]).lower()

        career_title = career.get("title", "")
        score = 0.0
        matches = 0

        for keyword, mapped_career in self.keyword_career_map.items():
            if keyword in user_text:
                if mapped_career == career_title:
                    score += 1.0
                    matches += 1

        
        career_keywords = [k.lower() for k in career.get("keywords", [])]
        for ck in career_keywords:
            if ck in user_text:
                score += 0.5

        
        return min(score / max(len(career_keywords) + 3, 1), 1.0)

    def _skill_score(self, profile: dict, career: dict) -> float:
        """Fraction of career's required skills the student already has."""
        student_skills = set(s.lower() for s in profile.get("skills", []))
        required = set(s.lower() for s in career.get("skills_required", []))
        if not required:
            return 0.0
        return len(student_skills & required) / len(required)

    def _trait_score(self, profile: dict, career: dict) -> float:
        """Personality trait overlap."""
        student = set(t.lower() for t in profile.get("personality_traits", []))
        career_traits = set(t.lower() for t in career.get("personality_traits", []))
        if not career_traits:
            return 0.0
        return len(student & career_traits) / len(career_traits)

    def _major_score(self, profile: dict, career: dict) -> float:
        """Bonus if student's major/degree directly maps to this career."""
        major = profile.get("major", "").lower()
        career_title = career.get("title", "")
        for key, careers in self.major_career_map.items():
            if key in major and career_title in careers:
                return 1.0
        return 0.0

    def _work_env_score(self, profile: dict, career: dict) -> float:
        """Bonus if work environment preference matches."""
        pref = profile.get("work_preference", "").lower()
        env = career.get("work_environment", "").lower()
        if not pref:
            return 0.5
        if pref in env or env in pref:
            return 1.0
        if "hybrid" in pref:
            return 0.7
        return 0.3

    #  Main recommendation function 

    def get_recommendations(self, profile: dict, top_n: int = 5) -> list:
        """
        Final score = weighted combination of 5 signals:
          40% keyword match  (most sensitive to interests/goals)
          20% SBERT semantic similarity
          20% skill overlap
          10% personality trait match
          10% major alignment + work env
        """
        # SBERT scores
        profile_text = self._profile_to_text(profile)
        profile_emb = self.model.encode([profile_text])
        sbert_scores = cosine_similarity(profile_emb, self.career_embeddings)[0]

        final_scores = []
        for i, career in enumerate(self.careers):
            kw    = self._keyword_score(profile, career)
            sbert = float(sbert_scores[i])
            skill = self._skill_score(profile, career)
            trait = self._trait_score(profile, career)
            major = self._major_score(profile, career)
            work  = self._work_env_score(profile, career)

            # Weighted final score
            final = (
                0.40 * kw    +
                0.20 * sbert +
                0.20 * skill +
                0.10 * trait +
                0.05 * major +
                0.05 * work
            )

            final_scores.append({
                "career": career,
                "match_score":        round(final * 100, 1),
                "semantic_score":     round(sbert * 100, 1),
                "skill_match":        round(skill * 100, 1),
                "trait_match":        round(trait * 100, 1),
                "keyword_match":      round(kw * 100, 1),
            })

        # Sort and take top N
        final_scores.sort(key=lambda x: x["match_score"], reverse=True)
        top_results = final_scores[:top_n]

        recommendations = []
        for rank, result in enumerate(top_results, 1):
            career = result["career"]
            recommendations.append({
                "rank":                rank,
                "career_id":           career["id"],
                "title":               career["title"],
                "match_score":         result["match_score"],
                "skill_match_percent": result["skill_match"],
                "trait_match_percent": result["trait_match"],
                "description":         career["description"],
                "skills_required":     career["skills_required"],
                "salary_range":        career["salary_range"],
                "job_outlook":         career["job_outlook"],
                "work_environment":    career["work_environment"],
                "industry":            career["industry"],
                "education_required":  career["education_required"],
                "explanation":         self._generate_explanation(profile, career, result),
            })

        return recommendations

    def _generate_explanation(self, profile: dict, career: dict, scores: dict) -> str:
        """Human-readable explanation of why this career was recommended."""
        reasons = []

        # Keyword match
        user_text = " ".join([
            " ".join(profile.get("interests", [])),
            " ".join(profile.get("skills", [])),
            profile.get("goals", ""),
        ]).lower()
        career_keywords = career.get("keywords", [])
        matched_kw = [k for k in career_keywords if k.lower() in user_text]
        if matched_kw:
            reasons.append(f"Your interest in '{matched_kw[0]}' strongly aligns with this field")

        # Skill match
        student_skills = set(s.lower() for s in profile.get("skills", []))
        required = career.get("skills_required", [])
        matched_skills = [s for s in required if s.lower() in student_skills]
        if matched_skills:
            reasons.append(f"Your skills in {', '.join(matched_skills[:3])} are directly required")

        # Trait match
        student_traits = set(t.lower() for t in profile.get("personality_traits", []))
        career_traits = career.get("personality_traits", [])
        matched_traits = [t for t in career_traits if t.lower() in student_traits]
        if matched_traits:
            reasons.append(f"Your {', '.join(matched_traits[:2])} personality fits this role well")

        # Major match
        major = profile.get("major", "").lower()
        subjects = [s.lower() for s in career.get("related_subjects", [])]
        if any(s in major or major in s for s in subjects):
            reasons.append(f"Your {profile.get('major', '')} background is directly relevant")

        if not reasons:
            reasons.append(
                f"Your overall profile has a {scores['match_score']}% compatibility with this career"
            )

        return ". ".join(reasons) + "."