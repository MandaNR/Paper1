import csv, random, json, math, itertools
from datetime import datetime, timedelta
import uuid

random.seed(42)

# Parameters
NUM_STUDENTS = 4000
NUM_ITEMS = 6000
NUM_SKILLS = 30
MIN_INTERACTIONS = 50
MAX_INTERACTIONS = 200

# Generate skills with prerequisites forming an acyclic graph
skills = []
for sid in range(NUM_SKILLS):
    prereq = []
    if sid > 0:
        prereq_count = random.randint(0, min(3, sid))
        prereq = random.sample(list(range(sid)), prereq_count)
    skills.append({
        "skill_id": sid,
        "skill_name": f"Skill_{sid}",
        "prerequisites": prereq
    })

# Save skill metadata
with open("skills_metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["skill_id", "skill_name", "prerequisites"])
    writer.writeheader()
    for s in skills:
        writer.writerow({
            "skill_id": s["skill_id"],
            "skill_name": s["skill_name"],
            "prerequisites": "|".join(map(str, s["prerequisites"]))
        })

# Item -> skills mapping (1-3 skills each)
item_skills = {}
for iid in range(NUM_ITEMS):
    k = random.randint(1, 3)
    item_skills[iid] = random.sample(range(NUM_SKILLS), k)

# Helper for response probability update per skill mastery
initial_mastery = {sid: random.uniform(0.1, 0.3) for sid in range(NUM_SKILLS)}
learning_rate = 0.05

rows = []

start_date = datetime(2023, 1, 1)

for student in range(NUM_STUDENTS):
    interactions = random.randint(MIN_INTERACTIONS, MAX_INTERACTIONS)
    current_time = start_date + timedelta(days=random.randint(0, 30))
    mastery = {sid: initial_mastery[sid] for sid in range(NUM_SKILLS)}
    last_timestamp = None
    session_id = uuid.uuid4().hex[:8]
    session_counter = 0

    for attempt in range(1, interactions + 1):
        if attempt % random.randint(5, 20) == 0:
            # new session
            session_id = uuid.uuid4().hex[:8]
            session_counter = 0
        session_counter += 1

        item = random.randint(0, NUM_ITEMS - 1)
        skills_for_item = item_skills[item]
        difficulty = random.uniform(0.2, 0.9)  # higher = harder

        # Estimate probability based on mastery average over associated skills and difficulty
        mastery_avg = sum(mastery[sid] for sid in skills_for_item) / len(skills_for_item)
        prob_correct = 0.2 + 0.6 * mastery_avg - 0.3 * difficulty
        prob_correct = max(0.05, min(prob_correct, 0.95))
        response = 1 if random.random() < prob_correct else 0

        # Update mastery (simple increase if correct)
        for sid in skills_for_item:
            if response == 1:
                mastery[sid] = min(1.0, mastery[sid] + learning_rate * (1 - mastery[sid]))
            else:
                mastery[sid] = max(0.0, mastery[sid] - 0.02)  # slight forgetting

        hint_used = 1 if (response == 0 and random.random() < 0.3) else 0

        # Timing
        gap_seconds = random.randint(30, 300) if last_timestamp else 0
        current_time += timedelta(seconds=gap_seconds)
        time_since_last = gap_seconds
        response_time_ms = int(random.gauss(3000 * difficulty + 500, 400))
        response_time_ms = max(500, response_time_ms)

        device_type = random.choices(["mobile", "laptop"], weights=[0.4, 0.6])[0]
        grade_level = random.choice(["K12", "Undergrad", "Master"])
        course_id = f"COURSE_{random.randint(1, 20)}"

        rows.append({
            "student_id": student,
            "item_id": item,
            "skill_ids": "|".join(map(str, skills_for_item)),
            "timestamp": current_time.isoformat(),
            "response": response,
            "attempt_number": attempt,
            "hint_used": hint_used,
            "time_since_last": time_since_last,
            "session_id": session_id,
            "response_time_ms": response_time_ms,
            "item_difficulty": round(difficulty, 3),
            "student_grade_level": grade_level,
            "course_id": course_id,
            "device_type": device_type
        })
        last_timestamp = current_time

# Write dataset
with open("synthetic_bdkt_dataset.csv", "w", newline="") as f:
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Basic stats
stats = {
    "num_rows": len(rows),
    "num_students": NUM_STUDENTS,
    "num_items": NUM_ITEMS,
    "num_skills": NUM_SKILLS,
}
with open("synthetic_bdkt_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("Dataset generation complete. Rows:", len(rows))
