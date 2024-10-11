# Example: Calculate the average minutes played per game
data['AvgMinutesPerGame'] = data['TotalMinutes'] / data['TotalGames']

np.random.seed(42)  
data['InjuryRisk'] = np.random.choice([0, 1], size=len(data), p=[0.7, 0.3])
data['GamesInLastMonth'] = np.random.randint(0, 5, size=len(data))  

# Calculate BMI
data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)

# Select features for modeling
features = ['Age', 'Position', 'AvgMinutesPerGame', 'GamesInLastMonth', 'PreviousInjuries', 'BMI']
X = data[features]
y = data['InjuryRisk']  # This is the target variable (1 = injured, 0 = not injured)
