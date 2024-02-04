import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from statsmodels.nonparametric.smoothers_lowess import lowess
from ipywidgets import interact

# Load data
data = pd.read_csv('data/Predicted Performance.csv')
win_rate = data.iloc[:, 0]
points = np.arange(len(win_rate))

# Function to update plot
def update(frac=0.05):
    lowess_fitted = lowess(win_rate, points, frac=frac)
    
    plt.figure(figsize=(10, 6))
    plt.plot(points, win_rate, alpha=0.3, label='Original Data')
    plt.plot(lowess_fitted[:, 0], lowess_fitted[:, 1], label='LOWESS Smoothed Curve', color='green')
    
    plt.xlabel('Point', fontproperties=times_font)
    plt.ylabel('Score', fontproperties=times_font)
    plt.title('Performance score of the player', fontproperties=times_font)
    plt.legend(prop=times_font)
    
    plt.show()

# Times New Roman font
times_font = FontProperties(family='Times New Roman', size=12)

# Interactive slider
interact(update, frac=(0.01, 0.99, 0.01))