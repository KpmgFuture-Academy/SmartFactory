import os
import json

MEMORY_PATH = "data/target_memory.json"
os.makedirs("data", exist_ok=True)

def check_existing_target(file_key):
    """
    Check if we already stored the target for this file_key.
    """
    if not os.path.exists(MEMORY_PATH):
        return None
    with open(MEMORY_PATH, "r") as f:
        memory = json.load(f)
    return memory.get(file_key)

def save_target_choice(file_key, target_col):
    """
    Save the user's target selection.
    """
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            memory = json.load(f)
    else:
        memory = {}

    memory[file_key] = target_col

    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)
    print(f"💾 Saved target column: '{target_col}' for file: {file_key}")

def ask_target_column(df, file_key):
    """
    Prompt user for main target column — or skip if none exists.
    """
    existing = check_existing_target(file_key)
    if existing:
        print(f"✅ Reusing previously selected target: {existing}")
        return existing

    print("\n❓ Is there a main target column in this dataset? (y/n)")
    choice = input("👉 Your answer: ").strip().lower()

    if choice != 'y':
        print("⏭️ Skipping target selection and feature engineering.")
        return None

    print("\n📌 Available columns in your data:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")

    while True:
        try:
            index = int(input("\n🔍 Enter the index of the main target column: "))
            target_col = df.columns[index]
            break
        except (ValueError, IndexError):
            print("❌ Invalid input. Please enter a valid column index.")

    save_target_choice(file_key, target_col)
    return target_col
