import re

# These functions are used in scrape.py

height_regex = re.compile(r'^(\d*)\' (\d*)"')
def height_inches(string):
    match = height_regex.search(string)
    return 12 * int(match.group(1)) + int(match.group(2))

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
