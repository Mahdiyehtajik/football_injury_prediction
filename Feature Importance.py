# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=[features[i] for i in indices])
plt.title('Feature Importances')
plt.show()
