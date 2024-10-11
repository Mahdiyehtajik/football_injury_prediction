# Check for missing values
print(data.isnull().sum())

# If necessary, fill or drop missing values (example)
data.fillna(0, inplace=True)

# Convert categorical variables to numerical ones (e.g., position)
data['Position'] = data['Position'].astype('category').cat.codes

# Ensure that all data is numeric and clean
print(data.info())
