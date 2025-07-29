import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

conn = sqlite3.connect("data/metrics.db")  # Replace with your database path

df = pd.read_sql_query("""
    SELECT metric_name, expected_number, value
    FROM raw_count_metrics
""", conn)

conn.close()

df['value'] = df['value'].astype(int)

# Group by metric_name and expected_number to compute accuracy
accuracy_df = (
    df.groupby(['metric_name', 'expected_number'])
      .agg(total=('value', 'count'), correct=('value', 'sum'))
      .reset_index()
)

# Compute accuracy percentage
accuracy_df['accuracy_pct'] = 100 * accuracy_df['correct'] / accuracy_df['total']

# Plot
plt.figure(figsize=(10, 6))
for object_class in accuracy_df['metric_name'].unique():
    subset = accuracy_df[accuracy_df['metric_name'] == object_class]
    object_class = object_class.split('_', 1)[-1]
    plt.plot(subset['expected_number'], subset['accuracy_pct'], marker='o', label=object_class)

plt.xlabel("Expected Number")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs expected number of objects (SD1.5)")
plt.legend(title="Object Class")
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=300)
