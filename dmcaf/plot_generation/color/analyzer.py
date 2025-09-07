import sqlite3
import os

def sort_dict_by_values(dictionary, desc=True):
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=desc))

model_names = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "SD-V1-5",
    "stabilityai/stable-diffusion-3.5-medium": "SD-3-5-Medium",
    "stabilityai/stable-diffusion-3-medium-diffusers": "SD-3-Medium",
    "stabilityai/stable-diffusion-2-1": "SD-2-1"
}

def calculate_average_color_accuracy_by_model(db_path, print_enabled=False):
    """
    Groups rows by model_name and calculates the average color_accuracy for each model.
    Args:
        db_path (str): Path to the SQLite database file.
    """
    ret = {}
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to group by model_name and calculate average color_accuracy
        query = """
        SELECT model_name, AVG(color_accuracy) AS avg_color_accuracy
        FROM image_evaluations
        GROUP BY model_name
        """
        cursor.execute(query)
        results = cursor.fetchall()

        if print_enabled:
            print("Model Name | Average Color Accuracy")
            print("----------------------------------")
        for row in results:
            if print_enabled:
                print(f"{model_names[row[0]]} | {row[1]:.2f}")
            ret[model_names[row[0]]] = row[1]
        return sort_dict_by_values(ret)

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        # Close the database connection
        if conn:
           conn.close()


def calculate_average_color_accuracy_by_confidence(db_path, print_enabled=False):
    """
    Calculates the average color_accuracy grouped by 5 color_confidence bins:
    1: 0.0-0.2, 2: 0.2-0.4, 3: 0.4-0.6, 4: 0.6-0.8, 5: 0.8-1.0
    Args:
        db_path (str): Path to the SQLite database file.
    """
    ret = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
        SELECT
            CASE
                WHEN color_confidence >= 0 AND color_confidence < 0.2 THEN '[0, 0.2)'
                WHEN color_confidence >= 0.2 AND color_confidence < 0.4 THEN '[0.2, 0.4)'
                WHEN color_confidence >= 0.4 AND color_confidence < 0.6 THEN '[0.4, 0.6)'
                WHEN color_confidence >= 0.6 AND color_confidence < 0.8 THEN '[0.6, 0.8)'
                WHEN color_confidence >= 0.8 AND color_confidence <= 1 THEN '[0.8, 1.0]'
            END AS confidence_bin,
            AVG(color_accuracy) AS avg_color_accuracy,
            COUNT(*) AS count
        FROM image_evaluations
        WHERE color_confidence IS NOT NULL AND color_accuracy IS NOT NULL
        GROUP BY confidence_bin
        ORDER BY confidence_bin ASC
        """
        cursor.execute(query)
        results = cursor.fetchall()

        if print_enabled:
            print("Confidence Bin | Average Color Accuracy | Count")
            print("-----------------------------------------------")
        for row in results:
            if print_enabled:
                print(f"{row[0]} | {row[1]:.2f} | {row[2]}")
            ret[row[0]] = row[1]
        return ret

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()


def calculate_average_color_accuracy_by_object(db_path, print_enabled=False):
    """
    Groups rows by model_name and object_name, and calculates the average color_accuracy
    and the count of rows for each combination.
    Args:
        db_path (str): Path to the SQLite database file.
    """
    ret = {}
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to group by model_name and object_name, calculate average color_accuracy, and count rows
        query = """
        SELECT target_object, AVG(color_accuracy) AS avg_color_accuracy, COUNT(*) AS count
        FROM image_evaluations
        WHERE detected_color IS NOT NULL
        GROUP BY target_object
        ORDER BY avg_color_accuracy DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()

        if print_enabled:
            print("Object Name | Average Color Accuracy | Count")
            print("----------------------------------------------------------")
        for row in results:
            if print_enabled:
                print(f"{row[0]} | {row[1]:.2f} | {row[2]}")
            ret[row[0]] = row[1]
        return ret

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        # Close the database connection
        if conn:
            conn.close()


def calculate_average_color_accuracy_for_low_color_variability_objects(db_path):
    objects = calculate_average_color_accuracy_by_object(db_path)

    low_color_variability = [
        "banana", "orange", "carrot", "apple", "fire hydrant", "traffic light", "zebra", "giraffe",
        "stop sign", "cow", "sheep", "bear", "cat", "dog", "horse", "elephant", "pizza", "hot dog", "sandwich",
        "wine glass", "mouse", "person"
    ]
    return {k: v for k, v in objects.items() if k in low_color_variability} 


def calculate_count_by_detected_and_expected_color(db_path, print_enabled=False):
    """
    Groups rows by (detected_color, expected_color) and counts the number of rows for each pair.
    Args:
        db_path (str): Path to the SQLite database file.
    """
    ret = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
        SELECT detected_color, expected_color, COUNT(*) AS count
        FROM image_evaluations
        WHERE detected_color IS NOT NULL AND expected_color IS NOT NULL AND expected_color != detected_color
        GROUP BY detected_color, expected_color
        ORDER BY count DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()

        if print_enabled:
            print("Detected Color | Expected Color | Count")
            print("--------------------------------------")
        total = 0
        for row in results:
            if print_enabled:
                print(f"{row[0]} | {row[1]} | {row[2]}")
            total += int(row[2])
            ret.append({
                'detected_color': row[0],
                'expected_color': row[1],
                'count': row[2]
            })
        if print_enabled:
            print(f"Total misclassifications: {total}")
        return ret

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()


def calculate_average_color_accuracy_by_pixel_ratio(db_path, print_enabled=False):
    """
    Calculates the average color_accuracy grouped by 5 pixel_ratio bins:
    1: 0.0-0.2, 2: 0.2-0.4, 3: 0.4-0.6, 4: 0.6-0.8, 5: 0.8-1.0
    Args:
        db_path (str): Path to the SQLite database file.
    """
    ret = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
        SELECT
            CASE
                WHEN pixel_ratio >= 0.0 AND pixel_ratio < 0.05 THEN '[0, 5)' 
                WHEN pixel_ratio >= 0.05 AND pixel_ratio < 0.10 THEN '[05, 10)'
                WHEN pixel_ratio >= 0.10 AND pixel_ratio < 0.15 THEN '[10, 15)'
                WHEN pixel_ratio >= 0.15 AND pixel_ratio < 0.20 THEN '[15, 20)'
                WHEN pixel_ratio >= 0.20 AND pixel_ratio < 0.25 THEN '[20, 25)'
                WHEN pixel_ratio >= 0.25 AND pixel_ratio < 0.30 THEN '[25, 30)'
                WHEN pixel_ratio >= 0.30 AND pixel_ratio < 0.35 THEN '[30, 35)'
                WHEN pixel_ratio >= 0.35 AND pixel_ratio < 0.40 THEN '[35, 40)'
                WHEN pixel_ratio >= 0.40 AND pixel_ratio < 0.45 THEN '[40, 45)'
                WHEN pixel_ratio >= 0.45 AND pixel_ratio < 0.50 THEN '[45, 50)'
                WHEN pixel_ratio >= 0.50 AND pixel_ratio <= 1 THEN '[50, 100)'
            END AS pixel_ratio_bin, model_name,
            AVG(color_accuracy) AS avg_color_accuracy,
            COUNT(*) AS count
        FROM image_evaluations
        WHERE pixel_ratio IS NOT NULL AND color_accuracy IS NOT NULL
        GROUP BY pixel_ratio_bin, model_name
        ORDER BY model_name ASC, pixel_ratio_bin ASC
        """
        cursor.execute(query)
        results = cursor.fetchall()

        if print_enabled:
            print("Model Name | Pixel Ratio Bin | Average Color Accuracy | Count")
            print("-----------------------------------------------")
        for row in results:
            if print_enabled:
                print(f"{model_names[row[1]]} | {row[0]} | {row[2]:.2f} | {row[3]}")
            if not model_names[row[1]] in ret:
                ret[model_names[row[1]]] = { 'accuracy': [], 'coverage': [] }
            ret[model_names[row[1]]]['coverage'].append(row[0])
            ret[model_names[row[1]]]['accuracy'].append(row[2])
        return ret

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()


# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "metrics-clip-complete.db")  # Update with the correct path if needed
    x = calculate_average_color_accuracy_by_pixel_ratio(db_path, print_enabled=True)
    print(x)