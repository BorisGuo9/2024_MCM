import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
file_path = 'data/importance_values_split.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Define the annotations for the labels
feature_annotations = {
    "x_1": "Number of games won in a set",
    "x_2": "Lead score in the current game",
    "x_3": "Whether the player is serving",
    "x_4": "Whether trailing in the current game",
    "x_5": "Number of sets lead in a match",
    "x_6": "Player 1's aces",
    "x_7": "Player 1's winners",
    "x_8": "Player 1's double faults",
    "x_9": "Player 1's unforced errors",
    "x_10": "Ratio of net points won to approaches",
    "x_11": "Ratio of break points won to opportunities",
    "x_12": "Total running distance in the match",
    "x_13": "Total running distance in the last three points",
    "x_14": "Running distance in the last point",
    "x_15": "Serving speed",
    "x_16": "Number of strokes per rally",
    "x_17": "Number of strokes in the rally"
}

# Ensure we have exactly 17 rows for x_1 to x_17
if len(data) >= 17:
    values = data.iloc[:17, 0]
    labels = [f"x_{i+1}" for i in range(17)]
else:
    # If less than 17 rows, pad with zeros and adjust labels accordingly
    labels = [f"x_{i+1}" for i in range(len(data))]
    values = data.iloc[:, 0]
    labels.extend([f"x_{i+len(data)+1}" for i in range(17 - len(data))])
    values = values.append(pd.Series([0] * (17 - len(data))))

# Create a color map
colors = plt.cm.get_cmap('viridis', len(labels))

# Create horizontal bar plot with different colors for each bar
plt.figure(figsize=(12, 10))
bars = plt.barh(labels, values, color=[colors(i) for i in range(len(labels))])

# Add the data values on the bars
for idx, bar in enumerate(bars):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
         f'{bar.get_width():.2f}',
         va='center', ha='left', fontname='Times New Roman')

plt.xlabel('Values', fontname='Times New Roman')
plt.ylabel('Features', fontname='Times New Roman')
plt.title('Feature Performance importance of split', fontname='Times New Roman')

plt.xticks(fontname='Times New Roman')
plt.yticks(fontname='Times New Roman')

# Create a legend
patch_list = [plt.Rectangle((0,0),1,1, color=colors(i), label=f'{label}: {feature_annotations[label]}') 
              for i, label in enumerate(labels)]
plt.legend(handles=patch_list, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
           fontsize='small', prop={'family': 'Times New Roman'})

# Show the plot with a legend
plt.tight_layout()
plt.savefig('figures/important_split.png', dpi=300, bbox_inches='tight')
# plt.show()