# Metis Week 2 Project: MMA Fight Duration Predictions

## Biff! Bam! Pow!

I scraped fighter and fight data from www.mixedmartialarts.com, the official database, in order to try to predict fight durations based on variables such as weight class and percentage of wins in previous fights. To make the data symmetrical with respect to both fighters, for variables that differed between fighters the sum or absolute value of difference was taken.

I grabbed the raw html (and a bunch of other stuff I didn't need) using `wget`:

```
wget -r -l0 www.mixedmartialarts.com/fighter/
```

I didn't make this into an executable script because it took a while to download all of the data (I left it running for more that 24 hours, although I'm not sure what the minimum time would have grabbed enough data), and I figured you wouldn't want to do that. The reason it required so much data is that the fighter pages come in no particular order, and getting complete data for a fight requires getting data from the fighter page of both fighters; the result was that I grabbed lots of data and then filtered it down later.

Python files:

- `scraping_helper_functions.py`
  - contains functions for parsing data in `scrape.py`
- `scrape.py`
  - attempts to parse fighter data from html (one page per fighter)
  - pickles the output for each fighter with enough parsable data
- `get_fights_helper_functions.py`
  - contains functions for:
    - aggregating each fighter's fights *before* each fight in question
    - generating statistics from each list of previous fights
- `get_fights_and_merge.py`
  - reads pickled fighter data, aggregating fights for which we have data for both fighters
  - merges in the data for the second fighter in each fight
  - generates columns for sums and differences of relevant stats
  - removes excess columns
  - pickles the final merged dataframe, as well as an intermediate dataframe for EDA
- `regress.py`
  - performs regression, generating stats and plots

I removed all the parts that eventually got dropped, including the postdiction features I cleaned before coming to my senses and some features I didn't quite manage to bugfix in time.

Cheers!
