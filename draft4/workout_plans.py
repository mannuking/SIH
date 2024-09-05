def get_workout_plans():
    """Returns a dictionary of available workout plans."""
    return {
        "Squats": {
            "name": "Squats",
            "target_reps": 20,
            "target_sets": 1,
            "instructions": "Stand with feet shoulder-width apart, lower your body as if sitting back into a chair, keep your chest up and knees over your toes.",
        },
        "Pushups": {
            "name": "Pushups",
            "target_reps": 20,
            "target_sets": 1,
            "instructions": "Start in a plank position, lower your body until your chest nearly touches the floor, then push back up.",
        },
    }
