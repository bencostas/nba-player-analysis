# -*- coding: utf-8 -*-
"""
CPS844 - A2

Association Analysis

Ben Costas 501025364
"""

import pandas as pd
from apyori import apriori

# Load the dataset with correct delimiter and encoding type
data = pd.read_csv("2023-2024 NBA Player Stats.csv", encoding='ISO-8859-1', delimiter=';')

# Remove duplicate players (only keep their TOT games)
data = data[data.duplicated('Player', keep=False) & data['Tm'].eq('TOT') | ~data.duplicated('Player', keep='first')]

# Only keep first position
data['Pos'] = data['Pos'].apply(lambda x: x.split('-')[0])

# Replace all 0s with 0.01 (for processing)
data = data.replace(0, 0.01)

# Keep players who played more than 20 games (assume players who played less games had little to no impact)
data = data[data['G'] >= 30]

# Combine Steals and BLocks to "STOCKS" --> measures defensive impact
data['STOCKS'] = data['STL'] + data['BLK']

# Combine Turnovers and Personal Fouls to "MISTAKES" --> measures the amount of mistakes the player makes per game 
data['MISTAKES'] = data['TOV'] + data['PF']

# Drop unused columns
data = data.drop(columns=['Player', 'Rk', 'Tm', 'G', 'GS', 'FG', 'FT', 'FTA', 'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF', '3P', '2P', 'FG%', '3PA', '2PA'])

# Discretize continuous attributes based on width (bins)
data['Age'] = pd.cut(data['Age'], bins=3, labels=["Developing", "Prime", "Near Retirement"])
data['MP'] = pd.cut(data['MP'], bins=3, labels=(['Low MP', 'Medium MP', 'High MP']))
data['FGA'] = pd.cut(data['FGA'], bins=3, labels=(['Low FGA', 'Medium FGA', 'High FGA']))
data['TRB'] = pd.cut(data['TRB'], bins=3, labels=(['Low TRB', 'Average TRB', 'High TRB']))

# Discretize continuous attributes based on frequency
points=[0, 5, 10, 16, 22, 36]
data['PTS'] = pd.cut(data['PTS'], bins=points, labels=(['Very Low PPG', 'Low PPG', 'Mid PPG', 'Star Level PPG', 'Elite PPG']))

threes=[0,0.20,0.33,0.385,1]
data['3P%'] = pd.cut(data['3P%'], bins=threes, labels=(['Non 3PT Shooter', 'Bad 3PT Shooter', 'Average 3PT Shooter', 'Elite 3PT Shooter']))

assists=[0,2,4,6,12]
data['AST'] = pd.cut(data['AST'], bins=assists, labels=(['Low AST', 'Mediocre AST', 'Average AST', 'Elite AST']))

twos=[0,0.5,0.55,1]
data['2P%'] = pd.cut(data['2P%'], bins=twos, labels=['Low 2P%', 'Average 2P%', 'Above Average 2P%'])

efg=[0,0.50,0.58,1]
data['eFG%'] = pd.cut(data['eFG%'], bins=efg, labels=['Below Average EFG', 'Average EFG', 'Above Average EFG'])

# Binarize continuous attributes
data['STOCKS'] = data['STOCKS'].apply(lambda x: 'High Defensive Impact' if x >= 1.5 else 'Low Defensive Impact' )
data['MISTAKES'] = data['MISTAKES'].apply(lambda x: 'High Mistakes' if x >= 3 else 'Low Mistakes')
data['FT%'] = data['FT%'].apply(lambda x: "Good FT Shooter" if x >= 0.80 else 'Bad FT Shooter')


# Now that the necessary data is preprocessed and discretized, we can perform apriori 
# To study the relation between the attributes

# Convert the DataFrame into a list of transactions
transactions = [row.dropna().tolist() for index, row in data.iterrows()]

# Perform Apriori algorithm
itemsets = apriori(transactions, min_support=0.15, min_confidence=0.90)

# Open a file in write mode
with open("association_rules_output.txt", "w") as file:
    rule_number = 1  # Initialize rule numbering
    for itemset in itemsets:
        for rule_index in range(len(itemset.ordered_statistics)):
            # Base (antecedent) of the rule
            base = list(itemset.ordered_statistics[rule_index].items_base)
            
            # Add (consequent) of the rule
            add = list(itemset.ordered_statistics[rule_index].items_add)
            
            # Support and confidence of the rule
            support = itemset.support
            confidence = itemset.ordered_statistics[rule_index].confidence
            
            # Formatting the output
            output = f"Rule #{rule_number}: {base} -> {add}\nSupport: {support}\nConfidence: {confidence}\n\n"
            file.write(output)
            
            rule_number += 1  # Increment rule number for the next rule
