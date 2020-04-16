import os
import pickle
import pandas as pd
from tqdm import tqdm as tq

pd.set_option('display.max_columns', None)

fighters_info_dir = os.path.relpath("./fighters_info")

with open("/home/james/Documents/metis/project_2/fighter_names") as f:
    names = f.read()
splt = names.split('\n')[:-1]

table = []
for name in tq(splt):
    with open(f"{fighters_info_dir}/{name}", 'rb') as f:
            fighter_info = pickle.load(f)
    past_fights_per_fight = get_past_fights_per_fight(fighter_info)
    fight_count = len(past_fights_per_fight)
    for fight, past_fights in past_fights_per_fight:
        # print(fight)
        if fight[1] in splt:
            table.append((
                name,
                fight[1],
                fight_count,
                fighter_info['height'],
                fighter_info['class'],
                fighter_info['male'],
                get_win_rate(past_fights),
                get_loss_rate(past_fights),
                get_fastest_win(past_fights),
                get_fastest_loss(past_fights),
                get_avg_win_time(past_fights),
                get_avg_loss_time(past_fights),
                fight[-2],
            ))
        # past_fight_ages_days = get_past_fight_ages_days(fighter_info)
        # past_fight_career_days = get_past_fight_career_days(past_fight_ages_days)
            # [print(past_fight, '\n') for past_fight in past_fights]
            # print(name)
table

len(table)

header = [
    'f1',
    'f2',
    'fight_count',
    'height',
    'class',
    'male',
    'win_rate',
    'loss_rate',
    'fastest_win',
    'fastest_loss',
    'avg_win_time',
    'avg_loss_time',
    'time',
]

table_in_lists = [list(tup) for tup in table]

df = pd.DataFrame(table_in_lists, columns=header)

# with open(f"/home/james/Documents/metis/project_2/second_dataframe.pickle", 'wb') as to_write:
    # pickle.dump(df, to_write)

with open(project_dir + "second_dataframe.pickle", 'rb') as to_read:
    df = pickle.load(to_read)

# small = df.iloc[0:500]
# small2 = df.iloc[0:500]

fights = df.copy(deep=True)

fighters = df.copy(deep=True)
del fighters['f2']
del fighters['time']
del fighters['male']
fighters.rename(columns = {
    'f1': 'f2',
    'fight_count': 'fight_count_2',
    'height': 'height_2',
    'class': 'class_2',
    'win_rate': 'win_rate_2',
    'loss_rate': 'loss_rate_2',
    'fastest_win': 'fastest_win_2',
    'fastest_loss': 'fastest_loss_2',
    'avg_win_time': 'avg_win_time_2',
    'avg_loss_time': 'avg_loss_time_2',
}, inplace=True)

fighters = fighters\
    .groupby(['f2'])\
    .agg({
        'fight_count_2': 'first',
        'height_2': 'first',
        'class_2': 'first',
        'win_rate_2': 'first',
        'loss_rate_2': 'first',
        'fastest_win_2': 'first',
        'fastest_loss_2': 'first',
        'avg_win_time_2': 'first',
        'avg_loss_time_2': 'first',
    })\
    .reset_index()

fights.columns, fighters.columns

merged = fights.merge(fighters, how='inner', on='f2')
merged['fight_count_dif'] = abs(merged['fight_count'] - merged['fight_count_2'])
merged['fight_count_sum'] = merged['fight_count'] + merged['fight_count_2']
merged['height_dif'] = abs(merged['height'] - merged['height_2'])
merged['height_sum'] = merged['height'] + merged['height_2']
merged['win_rate_dif'] = abs(merged['win_rate'] - merged['win_rate_2'])
merged['win_rate_sum'] = merged['win_rate'] + merged['win_rate_2']
merged['loss_rate_dif'] = abs(merged['loss_rate'] - merged['loss_rate_2'])
merged['loss_rate_sum'] = merged['loss_rate'] + merged['loss_rate_2']
merged['fastest_win_dif'] = abs(merged['fastest_win'] - merged['fastest_win_2'])
merged['fastest_win_sum'] = merged['fastest_win'] + merged['fastest_win_2']
merged['fastest_loss_dif'] = abs(merged['fastest_loss'] - merged['fastest_loss_2'])
merged['fastest_loss_sum'] = merged['fastest_loss'] + merged['fastest_loss_2']
merged['avg_win_time_dif'] = abs(merged['avg_win_time'] - merged['avg_win_time_2'])
merged['avg_win_time_sum'] = merged['avg_win_time'] + merged['avg_win_time_2']
merged['avg_loss_time_dif'] = abs(merged['avg_loss_time'] - merged['avg_loss_time_2'])
merged['avg_loss_time_sum'] = merged['avg_loss_time'] + merged['avg_loss_time_2']
del merged['f1']
del merged['f2']
del merged['class_2']
del merged['win_rate']
del merged['win_rate_2']
del merged['loss_rate']
del merged['loss_rate_2']
del merged['fastest_win']
del merged['fastest_win_2']
del merged['fastest_loss']
del merged['fastest_loss_2']
del merged['avg_win_time']
del merged['avg_win_time_2']
del merged['avg_loss_time']
del merged['avg_loss_time_2']
merged = merged.replace('0', np.NaN).dropna()

# with open(f"/home/james/Documents/metis/project_2/second_merged.pickle", 'wb') as to_write:
    # pickle.dump(merged, to_write)
