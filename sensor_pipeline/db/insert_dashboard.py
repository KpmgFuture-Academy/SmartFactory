from db.connection import get_connection
from datetime import datetime

def generate_dashboard_metrics(upload_id: str):
    """
    Aggregates and inserts a row of dashboard metrics using all data (no time filter).
    """

    conn, cur = get_connection()

    now = datetime.now()
    print(f"🕒 Aggregating metrics from all data up to: {now}")

    # 1. Average sensor value (Torque)
    cur.execute("""
        SELECT AVG((features->>'Torque [Nm]')::FLOAT)
        FROM input_sensor_data
        WHERE upload_id = %s
    """, (upload_id,))
    avg_value = cur.fetchone()[0] or 0.0

    # 2. Alarm count
    cur.execute("""
        SELECT COUNT(*)
        FROM prediction_result_sensor
        WHERE upload_id = %s
        AND predicted_failure = TRUE
    """, (upload_id,))
    alarm_count = cur.fetchone()[0] or 0

    # 3. First-time pass rate
    cur.execute("""
        SELECT COUNT(*) FILTER (WHERE anomaly_detected = FALSE) * 1.0 / NULLIF(COUNT(*), 0)
        FROM production_log
    """)
    pass_rate = cur.fetchone()[0] or 0.0

    # 4. Production count
    cur.execute("""
        SELECT COUNT(*) FROM production_log
    """)
    prod_count = cur.fetchone()[0] or 0

    # 5. Automation rate
    cur.execute("""
        SELECT COUNT(*) FILTER (WHERE is_automated) * 1.0 / NULLIF(COUNT(*), 0)
        FROM production_log
    """)
    auto_rate = cur.fetchone()[0] or 0.0

    # 6. MTTR
    cur.execute("""
        SELECT AVG(repair_duration_minutes)
        FROM mttr_log
    """)
    mttr = cur.fetchone()[0] or 0.0

    # 7. MTBF (placeholder)
    mtbf = 180.0

    # 8. Q-cost (placeholder)
    q_cost = alarm_count * 5.5

    print("✅ Computed values:")
    print("avg_value =", avg_value)
    print("alarm_count =", alarm_count)
    print("pass_rate =", pass_rate)
    print("prod_count =", prod_count)
    print("automation_rate =", auto_rate)
    print("mttr =", mttr)
    print("q_cost =", q_cost)

    # 9. Insert into dashboard_metrics
    cur.execute("""
        INSERT INTO dashboard_metrics (
            metric_time, sensor_id, avg_value, alarm_count,
            first_time_pass_rate, mtbf_minutes, mttr_minutes,
            q_cost, prod_per_hour, automation_rate
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        now, "S1", avg_value, alarm_count,
        pass_rate, mtbf, mttr,
        q_cost, prod_count, auto_rate
    ))

    conn.commit()
    cur.close()
    conn.close()
    print("📊 Inserted into dashboard_metrics ✅")
