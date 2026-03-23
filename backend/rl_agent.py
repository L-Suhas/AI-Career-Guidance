import json
import os

QTABLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtable.json")
ALPHA = 0.1

class RLAgent:
    def __init__(self):
        self.q_table = self._load_qtable()

    def _load_qtable(self):
        if os.path.exists(QTABLE_FILE):
            try:
                with open(QTABLE_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_qtable(self):
        with open(QTABLE_FILE, "w") as f:
            json.dump(self.q_table, f, indent=2)

    def update(self, career_title, action):
        current_q = self.q_table.get(career_title, 0.0)
        reward = 1.0 if action == "accept" else -1.0
        new_q = current_q + ALPHA * (reward - current_q)
        self.q_table[career_title] = round(new_q, 4)
        self._save_qtable()
        print(f"RL update: '{career_title}' {action} -> Q = {new_q:.4f}")

    def get_bonus(self, career_title):
        q = self.q_table.get(career_title, 0.0)
        return (q + 1.0) / 2.0 * 0.15

    def get_all_weights(self):
        return dict(self.q_table)

rl_agent = RLAgent()
