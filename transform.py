import re
from dateutil.relativedelta import relativedelta

date_regex = re.compile(r'^\d\d/\d\d/\d\d\d\d')
record_regex = re.compile(r'^(\d*)-(\d*)-(\d*).*')
height_regex = re.compile(r'^(\d*)\' (\d*)"')
years_months_days_regex = \
        re.compile(r'((\d*) years?)?(, )?((\d*) months?)?(, )?((\d*) days?)?')

def win_loss_tie(string):
    match = record_regex.search(string)
    return match.group(1), match.group(2), match.group(3)

def height_inches(string):
    match = height_regex.search(string)
    return 12 * int(match.group(1)) + int(match.group(2))

def percent(string):
    return float(string[:-1]) / 100

def seconds(string):
    # For fight lengths
    # Fights can't be longer than 25 min by Unified Rules
    if ':' in string and len(string.split(':')) == 2:
        minutes, seconds = string.split(':')
        return 60 * int(minutes) + int(seconds)
    elif string:
        # For round numbers
        return 5 * 60 * int(string)
    else:
        return 0

def win(table_data):
    if table_data.find("span", class_="Loss"):
        return 1
    elif table_data.find("span", class_="Win"):
        return 3
    elif table_data.find("span", class_="NC"):
        return 2
    elif table_data.find("span", class_="Draw"):
        return 2
    elif table_data.find("span", class_="TBD"):
        return 2
    else:
        return 0

def days(string):
    match = years_months_days_regex.search(string)
    years = match.group(2) or 0
    months = match.group(5) or 0
    days = match.group(8) or 0
    return 365 * int(years) + 30 * int(months) + int(days)

weight_classes = {
    "atomweight": 105,
    "strawweight": 115,
    "flyweight": 125,
    "bantamweight": 135,
    "featherweight": 145,
    "lightweight": 155,
    "super-lightweight": 165,
    "welterweight": 170,
    "super-welterweight": 175,
    "middleweight": 185,
    "super-middleweight": 195,
    "light-heavyweight": 205,
    "cruiserweight": 225,
    "heavyweight": 265,
    "super-heavyweight": 280
}

def weight_class_to_pounds(string):
    return weight_classes[string.lower()]

def transform_fighter_info(info):
    if 'Height' in info:
        info['height'] = height_inches(info['Height'])
        del info['Height']
    if 'Gender' in info:
        info['male'] = info['Gender'] == 'Male'
        del info['Gender']
    if 'Weight Class' in info:
        info['class'] = weight_class_to_pounds(info['Weight Class'])
        del info['Weight Class']

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

date_when_scraped = datetime.datetime.strptime('04/10/2020', '%m/%d/%Y')
def get_past_fight_ages_days(fighter_info, past_fights):
    age_when_scraped = fighter_info['age']
    estimated_birthday = \
            date_when_scraped - relativedelta(years=age_when_scraped)
    for fight in past_fights[0]:
        print(fight, "\n\n")
    print(past_fights[0])
    return [(fight[2] - estimated_birthday).days for fight in past_fights]
# x = get_past_fights_per_fight(fighters_info["Aaron-Garcia:C166F408B4C02EB0"])
# pfad = get_past_fight_ages_days(
        # fighters_info["Aaron-Garcia:C166F408B4C02EB0"],
        # x
    # )

def get_past_fight_career_days(past_fight_ages_days):
    first_fight_age = \
            0 if not past_fight_ages_days else min(past_fight_ages_days)
    return [age - first_fight_age for age in past_fight_ages_days]
