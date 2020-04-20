def get_past_fights_per_fight(fighter_info):
    """
    Takes in a fighter_info dict, which contains a bunch of stats,
        of which we use the 'fights' field, which is a list of fight records.
    Returns a list of tuples, one for each fight record in the input.
    The first element in each tuple is the original fight record.
    The second element in each tuple is a list of fight records of past fights.
    Each fight record is a tuple containing these fields:
        fighter, opponent, date, time, outcome
    """
    fights = fighter_info['fights']
    transformed_fights = []
    for fight in fights:
        fighter, opponent, date, duration, outcome = fight
        past_fights = [(_fighter, _opponent, _date, _duration, _outcome) \
                for (_fighter, _opponent, _date, _duration, _outcome) \
                in fights if _date < date]
        transformed_fights = transformed_fights + [(fight, past_fights)]
    return transformed_fights

def get_win_rate(past_fights):
    fights = len(past_fights)
    wins = len([fight for fight in past_fights if fight[4] == 3])
    return 0 if not fights else wins / fights

def get_loss_rate(past_fights):
    fights = len(past_fights)
    losses = len([fight for fight in past_fights if fight[4] == 1])
    return 0 if not fights else losses / fights

def get_fastest_win(past_fights):
    win_times = [fight[3] for fight in past_fights if fight[4] == 3]
    return 0 if not win_times else min(win_times)

def get_fastest_loss(past_fights):
    loss_times = [fight[3] for fight in past_fights if fight[4] == 1]
    return 0 if not loss_times else min(loss_times)

def get_avg_win_time(past_fights):
    win_times = [fight[3] for fight in past_fights if fight[4] == 3]
    return 0 if not win_times else sum(win_times) / len(win_times)

def get_avg_loss_time(past_fights):
    loss_times = [fight[3] for fight in past_fights if fight[4] == 1]
    return 0 if not loss_times else sum(loss_times) / len(loss_times)
