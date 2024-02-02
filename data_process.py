import csv
import os
from typing import List, Dict
import matplotlib.pyplot as plt

def parse_match_id(match_id: str) -> (int, int):
    parts = match_id.split('-')
    round_no = int(parts[2][1])  # Extracting the round number
    match_no = int(parts[2][2:])  # Extracting the match number
    return round_no, match_no

def parse_time(time_str: str) -> int:
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def parse_score(score: str) -> int:
    score_mapping = {'0': 'Love', '15': 15, '30': 30, '40': 40, 'AD': 'Advantage'}
    return score_mapping.get(score, score)

def parse_boolean(value: str) -> bool:
    return value == '1'

def parse_shot_type(shot: str) -> str:
    return 'Backhand' if shot == 'B' else 'Forehand' if shot == 'F' else None

def parse_speed(speed_str: str) -> int:
    return None if speed_str == 'NA' else int(speed_str)

def parse_serve_width(width: str) -> str:
    width_mapping = {
        'B': 'Body', 
        'BC': 'Body/Center', 
        'BW': 'Body/Wide', 
        'C': 'Center', 
        'W': 'Wide'
    }
    return width_mapping.get(width, width)

def parse_depth(depth: str) -> bool:
    return depth == 'CTL'#是否Close to Line

def parse_return_depth(depth: str) -> bool:
    return None if depth == 'NA' else depth == 'D'# 是否Deep


def read_csv_to_dict(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        processed_data = []
        for row in csv_reader:
            row['round_no'], row['match_no'] = parse_match_id(row['match_id'])
            row['elapsed_time'] = parse_time(row['elapsed_time'])
            for key in ['set_no', 'game_no', 'point_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games']:
                row[key] = int(row[key])
            for key in ['p1_score', 'p2_score']:
                row[key] = parse_score(row[key])
            for key in ['server', 'serve_no', 'point_victor', 'game_victor', 'set_victor']:
                row[key] = int(row[key])
            for key in ['p1_points_won', 'p2_points_won','rally_count']:
                row[key] = int(row[key])
            for key in ['p1_ace', 'p2_ace', 'p1_winner', 'p2_winner', 'p1_net_pt', 'p2_net_pt', 
                        'p1_net_pt_won', 'p2_net_pt_won', 'p1_unf_err', 'p2_unf_err', 'p1_break_pt', 
                        'p2_break_pt', 'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 
                        'p2_break_pt_missed']:
                row[key] = parse_boolean(row[key])
            row['winner_shot_type'] = parse_shot_type(row['winner_shot_type'])
            for key in ['p1_distance_run', 'p2_distance_run']:
                row[key] = float(row[key])
            for key in ['speed_mph']:
                row[key] = parse_speed(row['speed_mph'])
            for key in ['serve_width']:
                row[key] = parse_serve_width(row['serve_width'])
            for key in ['serve_depth']:
                row[key] = parse_depth(row['serve_depth'])
            for key in ['return_depth']:
                row[key] = parse_return_depth(row['return_depth'])
            processed_data.append(row)
        return processed_data


csv_data = read_csv_to_dict('data/Wimbledon_featured_matches.csv')
# for row in csv_data[3:4]:  # 注意，列表索引从0开始
#     print(row)

def clean_data(data: List[Dict[str, any]]) -> List[Dict[str, any]]:
    processed_list = []
    for row in data:
        identifier = f"{row['round_no']}-{row['match_no']}-{row['set_no']}-{row['game_no']}-{row['point_no']}"
        new_entry = {
            # 'identifier': identifier,
            'match_id': '-'.join(identifier.split('-')[0:2]),
            # 'point_all': identifier.split('-')[4],
            'point_p1': row['p1_points_won'],
            'point_p2': row['p2_points_won'],
            'time': row['elapsed_time']
        }
        processed_list.append(new_entry)

    return processed_list

new_data = clean_data(csv_data)
for row in new_data[5:10]:  # 注意，列表索引从0开始
    print(row)
    

def plot_match_time_series(data, font='Times New Roman', point_size=2, save_dir='figures'):
    r"""Plots time series for each match with 'point' as the y-axis and saves the figures.

    Args:
        data (List[Dict[str, any]]): The processed list of dictionaries containing identifiers and time.
        font (str): The font family to use for text in the plot.
        point_size (int): The size of the points in the plot.
        save_dir (str): Directory to save the figures.
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Group data by match
    matches = {}
    for item in data:
        match_id = item['identifier'].split('-')[0:2]  # Extracting round and match number
        match_key = '-'.join(match_id)
        if match_key not in matches:
            matches[match_key] = []
        matches[match_key].append(item)

    # Plotting each match
    for match, items in matches.items():
        plt.figure(figsize=(12, 8))  # Increased figure size for better readability
        times = [item['time'] for item in items]
        points = [int(item['identifier'].split('-')[-1]) for item in items]

        # Calculate label intervals
        x_label_interval = max(1, (max(times) - min(times)) // 10)
        y_label_interval = max(1, (max(points) - min(points)) // 10)

        plt.plot(times, points, marker='o', markersize=point_size, linestyle='-')
        plt.xlabel('Elapsed Time (seconds)', fontsize=14, fontname=font)
        plt.ylabel('Point', fontsize=14, fontname=font)
        plt.title(f'Time Series for Match {match}', fontsize=16, fontname=font)

        # Set x-axis and y-axis labels with dynamic intervals
        plt.xticks(range(min(times), max(times) + 1, x_label_interval), fontsize=12, fontname=font)
        plt.yticks(range(min(points), max(points) + 1, y_label_interval), fontsize=10, fontname=font)
        
        plt.grid(True)
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(save_dir, f'time_series_{match}.png'))
        plt.close()
        
# plot_match_time_series(new_data)

def plot_match_trends(data, font='Times New Roman', save_dir='figures'):
    r"""Plots the match trends using a multiple line chart and saves the figures.

    Args:
        data (List[Dict[str, any]]): The processed list of dictionaries containing match data.
        font (str): The font family to use for text in the plot.
        save_dir (str): Directory to save the figures.
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Group data by match
    matches = {}
    for item in data:
        match_id = item['match_id']
        if match_id not in matches:
            matches[match_id] = {'time': [], 'point_p1': [], 'point_p2': []}
        matches[match_id]['time'].append(item['time'])
        matches[match_id]['point_p1'].append(item['point_p1'])
        matches[match_id]['point_p2'].append(item['point_p2'])

    # Plotting each match
    for match, points in matches.items():
        plt.figure(figsize=(10, 6))
        plt.plot(points['time'], points['point_p1'], label='Player 1', marker='o')
        plt.plot(points['time'], points['point_p2'], label='Player 2', marker='o')
        plt.xlabel('Elapsed Time (seconds)', fontsize=14, fontname=font)
        plt.ylabel('Points', fontsize=14, fontname=font)
        plt.title(f'Match Trend for {match}', fontsize=16, fontname=font)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12, fontname=font)
        plt.yticks(fontsize=12, fontname=font)
        plt.grid(True)
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(save_dir, f'match_trend_{match}.png'))
        plt.close()


# plot_match_trends(new_data)