import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the CSV (relative path)
df = pd.read_csv("roundup/loglin_roundup1.csv")

print(df.columns)

# Create the rate variable
df['rate'] = df['articles with log DV'] / df['Articles']


# Drop rows with missing rate values
df = df.dropna(subset=['rate'])


# Create bar plot with proper positioning
plt.figure(figsize=(14, 6))
x_pos = range(len(df))
plt.bar(x_pos, df['rate'], alpha=0.6, color='steelblue')
plt.ylabel('Log-Linear DV articles/Total articles')
plt.title('Prop. of Articles with log-linear DVs by Journal')
plt.xticks(ticks=x_pos, labels=df['Journal'], rotation=45, ha='right', fontsize=7)
plt.ylim(0, 1)


# Add note at the bottom
plt.figtext(0.5, 0.01, 'Note: Top 5 journals we used a census of articles from all issues published in 2020, for all other journals we computed this from all articles published in the first two issues of 2020. \n Comments and special ediitions not included.', 
            ha='center', fontsize=6, style='italic')

plt.tight_layout()

# Save the figure (relative path)
plt.savefig("roundup/loglin_roundupgraph.png", dpi=300, bbox_inches='tight')

plt.show()