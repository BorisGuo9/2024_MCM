from data_process import read_csv_to_dict
import matplotlib.pyplot as plt
import networkx as nx
import textwrap

csv_data = read_csv_to_dict('data/Wimbledon_featured_matches.csv')
# print(csv_data)
def wrap_labels(labels, wrap_length):
    return {key: textwrap.fill(value, wrap_length) for key, value in labels.items()}

def calculate_victors(csv_data):
    for match_id in {item['match_id'] for item in csv_data}:
        match_data = [item for item in csv_data if item['match_id'] == match_id]

        # Sort the data by set_no, game_no, and point_no to get the last point in each game
        match_data.sort(key=lambda x: (x['set_no'], x['game_no'], x['point_no']))

        last_game_score = {}
        last_set_score = {}
        for point in match_data:
            game_key = (point['set_no'], point['game_no'])
            set_key = point['set_no']

            # Determine game victor
            if game_key not in last_game_score or point['point_no'] > last_game_score[game_key][2]:
                p1_points = point['p1_points_won']
                p2_points = point['p2_points_won']
                if p1_points >= 4 and p1_points - p2_points >= 2:
                    point['game_victor'] = 1
                elif p2_points >= 4 and p2_points - p1_points >= 2:
                    point['game_victor'] = 2
                else:
                    point['game_victor'] = None
                last_game_score[game_key] = (point['game_victor'], point['point_no'], point['point_no'])

            # Determine set victor
            if set_key not in last_set_score or point['game_no'] > last_set_score[set_key][1]:
                p1_games = point['p1_games']
                p2_games = point['p2_games']
                if p1_games >= 6 and p1_games - p2_games >= 2:
                    point['set_victor'] = 1
                elif p2_games >= 6 and p2_games - p1_games >= 2:
                    point['set_victor'] = 2
                else:
                    point['set_victor'] = None
                last_set_score[set_key] = (point['set_victor'], point['game_no'])

        # Determine match victor based on the set victories
        p1_sets_won = sum(1 for point in match_data if point.get('set_victor') == 1)
        p2_sets_won = sum(1 for point in match_data if point.get('set_victor') == 2)
        if p1_sets_won > p2_sets_won:
            match_victor = 1
        elif p2_sets_won > p1_sets_won:
            match_victor = 2
        else:
            match_victor = None
        for point in match_data:
            point['match_victor'] = match_victor

    return csv_data

# Call the function with your data
csv_data = calculate_victors(csv_data)
# print(csv_data)

def draw_tennis_match_network(data):
    G = nx.DiGraph()  # 使用有向图

    for match in data:
        player1 = match['player1']
        player2 = match['player2']
        round_match_label = f"{match['round_no']}-{match['match_no']}"
        point_victor = match['match_victor']

        # 根据point_victor的值确定边的方向
        if point_victor == 1:
            G.add_edge(player1, player2, label=round_match_label)
        else:
            G.add_edge(player2, player1, label=round_match_label)

    # 在绘制网络图之前先设置好布局和样式
    pos = nx.spring_layout(G, k=0.2)  # 'k' 参数控制节点之间的距离

    # 准备节点标签，将名字分为两行显示
    wrapped_labels = wrap_labels({n: n for n in G.nodes()}, 10)  # 每10个字符换行

    # 绘制节点和边，调整字体大小
    nx.draw(G, pos, labels=wrapped_labels, with_labels=True, node_color='lightblue', node_size=800, font_size=6, arrowsize=20)

    # 绘制边标签，并调整标签位置以防止重叠
    edge_labels = nx.get_edge_attributes(G, 'label')
    label_pos = 0.3  # 标签位置在边的中间
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=label_pos)

    plt.show()

# draw_tennis_match_network(csv_data)