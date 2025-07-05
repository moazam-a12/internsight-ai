import pandas as pd
import random

def generate_intern_data(num_records=12000, start_id=301):
    data = []

    for i in range(num_records):
        intern_id = start_id + i
        task_completion = round(random.uniform(1.0, 16.5), 2)
        feedback = round(random.uniform(1.0, 5.0), 1)
        attendance = round(random.uniform(0.0, 100.0), 2)

        # Continuous performance score (0 to 1)
        # Lower task completion time and higher feedback = better performance
        norm_task = 1 - (task_completion - 1.0) / (16.5 - 1.0)  # Normalize to [0, 1], lower is better
        norm_feedback = (feedback - 1.0) / (5.0 - 1.0)  # Normalize to [0, 1], higher is better
        performance = round(0.5 * norm_task + 0.5 * norm_feedback, 2)  # Weighted average

        data.append([
            intern_id,
            task_completion,
            feedback,
            attendance,
            performance
        ])

    df = pd.DataFrame(data, columns=[
        "Intern_ID",
        "Task_Completion_Time_Hrs",
        "Feedback_Rating",
        "Attendance_Percentage",
        "Performance"
    ])

    return df

# Generate dataset
df = generate_intern_data(num_records=12000)
df.to_csv("intern_performance.csv", index=False)
print("Dataset saved as 'intern_performance.csv'")