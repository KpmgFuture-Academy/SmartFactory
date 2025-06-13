import pandas as pd
import numpy as np
import os
import json

MEMORY_PATH = "data/sensor_limits.json"

def ask_user_manual_limits(df):
    print("\n📊 Available numeric columns:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for i, col in enumerate(numeric_cols):
        print(f"{i}: {col}")

    use_manual = input("\n❓ Do you want to define sensor limits manually? (y/n): ").strip().lower()
    manual_limits = {}

    if use_manual != "y":
        return {}, numeric_cols  # Skip to auto

    selected = input("🔍 Enter indices of columns to define manually (comma-separated): ").strip()
    selected_indexes = [int(x.strip()) for x in selected.split(",") if x.strip().isdigit()]
    selected_cols = [numeric_cols[i] for i in selected_indexes if 0 <= i < len(numeric_cols)]

    for col in selected_cols:
        min_val = float(input(f"🔧 Enter MIN value for {col}: ").strip())
        max_val = float(input(f"🔧 Enter MAX value for {col}: ").strip())
        manual_limits[col] = {"min": min_val, "max": max_val}

    return manual_limits, [col for col in numeric_cols if col not in manual_limits]

def auto_define_limits(df, remaining_cols):
    auto_limits = {}
    for col in remaining_cols:
        q_low, q_high = df[col].quantile(0.05), df[col].quantile(0.95)
        auto_limits[col] = {"min": q_low, "max": q_high}
    return auto_limits

def save_sensor_limits(file_key, manual, auto):
    full_limits = {
        "manual": manual,
        "auto": auto
    }

    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            memory = json.load(f)
    else:
        memory = {}

    memory[file_key] = full_limits
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)

def define_sensor_limits(df, file_key):
    manual_limits, auto_cols = ask_user_manual_limits(df)
    auto_limits = auto_define_limits(df, auto_cols)
    save_sensor_limits(file_key, manual_limits, auto_limits)
    return {**manual_limits, **auto_limits}

def apply_limits(df, limits):
    # ⛔ Do NOT clip or modify values. Just return as-is.
    return df
