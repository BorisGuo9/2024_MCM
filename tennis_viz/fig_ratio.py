from data_process import read_csv_to_dict
import pandas as pd
import matplotlib.pyplot as plt
import os

csv_data = read_csv_to_dict('data/Wimbledon_featured_matches.csv')
data_df = pd.DataFrame(csv_data)
for row in csv_data[3:4]:  
    print(row)

def draw_tennis_match_plots_with_step_fit(data_df, window_size=30):
    # Make sure the 'figures' directory exists
    os.makedirs('figures', exist_ok=True)
    
    for round_no in data_df['round_no'].unique():
        round_data = data_df[data_df['round_no'] == round_no]
        
        for match_no in round_data['match_no'].unique():
            match_data = round_data[round_data['match_no'] == match_no]
            plt.figure(figsize=(14, 7))

            # Set the font to Times New Roman
            plt.rcParams["font.family"] = "Times New Roman"
            
            # 绘制得分比率折线图
            player1_ratio = match_data['p1_points_won'] / match_data['p2_points_won']
            player2_ratio = match_data['p2_points_won'] / match_data['p1_points_won']
            plt.plot(match_data['point_no'], player1_ratio, label='Player 1', alpha=0.3)
            plt.plot(match_data['point_no'], player2_ratio, label='Player 2', alpha=0.3)

            # 计算移动平均拟合线
            player1_fit = player1_ratio.rolling(window=window_size, min_periods=1).mean()
            player2_fit = player2_ratio.rolling(window=window_size, min_periods=1).mean()
            plt.step(match_data['point_no'], player1_fit, label='Player 1 Fit', where='mid', color='red')
            plt.step(match_data['point_no'], player2_fit, label='Player 2 Fit', where='mid', color='blue')

            plt.xlabel('Point Number')
            plt.ylabel('Points Ratio')
            plt.title(f'Points Won Ratio Over Time - Round {round_no} Match {match_no}')
            plt.legend()
            
            # Save the figure
            figure_filename = f'figures/Round_{round_no}_Match_{match_no}.png'
            plt.savefig(figure_filename, format='png')
            
            # Clear the current figure to free memory for the next plot
            plt.clf()

# The function call is commented out since it's provided as an example and won't be executed here
draw_tennis_match_plots_with_step_fit(data_df, window_size=30)