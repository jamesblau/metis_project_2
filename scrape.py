import os
import pickle
from bs4 import BeautifulSoup

fighter_dir = "www.mixedmartialarts.com/fighter/"

# This directory cointains many hmtl files
# We loop through, attempting to find the fields we need
# If we find them, we parse the html and pickle the result
# Many of the files don't contain all desired fields, so we ignore them

for name in os.listdir(fighter_dir):
    # Ignore fighters with weird URLs
    if name.split(':')[0]:
        try:
            with open(fighter_dir + name) as to_read:
                html = to_read.read()
        except:
            pass
    else:
        html = ""
    # This entire block deals with the data from one fighter's page
    if html:
        info = {}
        fights = []
        soup = BeautifulSoup(html, 'lxml')
        # Find all data in key/value format
        # Though we can unfortunately only use a few such pairs
        for table in soup.find_all('table', class_="fighter-info"):
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                heads = [ele.text.strip() for ele in row.find_all('th')]
                cols = [ele.text.strip() for ele in row.find_all('td')]
                for head, col in zip(heads, cols):
                    if head and col and col != '-':
                        info[head] = col
        # Separately grab a list for each feature of previous fights
        #   (date, duration, opponent, outcome, etc.)
        #   then combine them into a list of past fights for this fighter
        for table in soup.find_all('table', class_="fighter-record"):
            links = [link.attrs['href'] for link in table.find_all('a')]
            opponents = [link.split('fighter/')[1] \
                    for link in links if 'fighter/' in link]
            fighter_col = [name for opponent in opponents]
            dates = [datetime.datetime.strptime(date.text.strip(), '%m/%d/%Y') \
                    for date in table.find_all('td') \
                    if date_regex.match(date.text.strip())]
            rounds = [t.text.strip() \
                    for t in table.find_all('td', class_="text-center")]
            last_round_times = [t.text.strip() \
                    for t in table.find_all('td', class_="text-right")]
            times = [seconds(rnd) + seconds(last_round_time) \
                    for (rnd, last_round_time) \
                    in zip(rounds, last_round_times)]
            outcomes = [win(maybe_outcome) \
                    for maybe_outcome in table.find_all('td') \
                    if win(maybe_outcome)]
            # In testing I checked that the lengths matched up
            #   but then disabled these for speed
            # assert(len(opponents) == len(dates))
            # assert(len(opponents) == len(rounds))
            # assert(len(opponents) == len(last_round_times))
            # assert(len(opponents) == len(times))
            # assert(len(opponents) == len(outcomes))
            fights = fights + \
                    list(zip(fighter_col, opponents, dates, times, outcomes))
        # Only parse and write the final output if we have certain fields
        if all([key in info for key in ['Height', 'Weight Class', 'Gender']]):
            info['fights'] = fights
            transform_fighter_info(info)
            # Write the info for each fighter
            with open("fighters_info/" + name, 'wb') as to_write:
                pickle.dump(info, to_write)
            # Separately keep a list of names, so that we can later
            #   filter out fights against fighters without sufficient data
            with open("fighter_names", 'a') as names_file:
                names_file.write(name + '\n')
