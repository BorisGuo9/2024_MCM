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
    return depth == 'CTL'

def parse_return_depth(depth: str) -> bool:
    return None if depth == 'NA' else depth == 'D'


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