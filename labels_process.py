from data_process import read_csv_to_dict
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("data/Wimbledon_featured_matches.csv")
data.loc[(data.serve_depth=='CTL'),'serve_depth'] = 1
data.loc[(data.serve_depth=='NCTL'),'serve_depth'] = 0
data.loc[(data.return_depth=='D'),'return_depth'] = 1
data.loc[(data.return_depth=='ND'),'return_depth'] = 0
data.loc[(data.p1_score=='AD'),'p1_score'] = 50
data.loc[(data.p2_score=='AD'),'p2_score'] = 50
data['p1_score'] = data['p1_score'].astype(int)
data['p2_score'] = data['p2_score'].astype(int)



def label_proc(data,label_vic,label_no):
    no = 0
    count = 0
    for row in range(no, data.shape[0]):   #data.shape[0]
        if row == 7283:
            for index in range(no-1 , no+count):
                    # print(no+count)
                    data.loc[index, label_vic] = data[label_vic][no+count]
                    # data[label_vic][index] = data[label_vic][no+count]
        else:
            if data[label_no][row] == data[label_no][no]:
                count+= 1
            else:
                for index in range(no-1 , no+count-1):
                    data.loc[index, label_vic] = data[label_vic][no+count-1]
                no = no+count+1
                count = 0

# label_proc(data,'game_victor','game_no')
# data = pd.read_csv("Wimbledon_with_victor_labels.csv")


# label_proc(data,'set_victor','set_no')
# df = pd.DataFrame(data)
# df.to_csv("Wimbledon_with_victor_labels_2.csv",index=False)

data = read_csv_to_dict("Wimbledon_with_victor_labels_2.csv")

for match_id in {item['match_id'] for item in data}:
    match_data = [item for item in data if item['match_id'] == match_id]

        # Sort the data by set_no, game_no, and point_no to get the last point in each game
    match_data.sort(key=lambda x: (x['set_no'], x['game_no'], x['point_no']))

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




