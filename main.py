"""
L'objectif de ce projet est d'analyser les données de ventes de jeux vidéo et les notes des jeux afin de répondre à deux questions principales :


1.   **Impact des Notes sur les Ventes** : Déterminer si les notes des jeux (MetaCritic et utilisateurs) influencent les ventes globales.

2.  **Différences Régionales des Ventes** : Explorer les variations des ventes de jeux vidéo par région (Amérique du Nord, Europe, Japon, autres régions).

"""

# Import des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Chargement des données

# Données de ventes de jeux vidéo
games_sales_url = "https://g-schmit.github.io/data/vg_sales.csv"

# Données de notes de Metacritic
meta_games_url = "https://g-schmit.github.io/data/all_meta_games.csv"

# Données de jeux bannis
banned_games_url = "/content/drive/MyDrive/Banned_Games.csv"

# Données de jeux polulaires
popular_games_url = "/content/drive/MyDrive/Popular_Games.csv"

# Chargement des fichiers CSV dans des DataFrames
games_sales_df = pd.read_csv(games_sales_url)
meta_games_df = pd.read_csv(meta_games_url)
banned_games_df = pd.read_csv(banned_games_url)
popular_games_df = pd.read_csv(popular_games_url)

# Exploration des données

# 1. Aperçu des données

# Afficher les premières lignes des dataframes
print(games_sales_df.head())
print(meta_games_df.head())
print(banned_games_df.head())
print(popular_games_df.head())

# Afficher les informations des dataframes
print(games_sales_df.info())
print(meta_games_df.info())
print(banned_games_df.info())
print(popular_games_df.info())

# Afficher les descriptions statistiques des dataframes
print(games_sales_df.describe())
print(meta_games_df.describe())
print(banned_games_df.describe())
print(popular_games_df.describe())

# 2. Visualiser les données

# 2.1 Distributions des ventes

# Distribution des ventes en Amérique du Nord
sns.histplot(games_sales_df['NA_Sales'], bins=30, kde=True)
plt.title('Distribution des ventes en Amérique du Nord')
plt.xlabel('Ventes en millions')
plt.ylabel('Fréquence')
plt.show()

# Distribution des ventes en Europe
sns.histplot(games_sales_df['EU_Sales'], bins=30, kde=True)
plt.title('Distribution des ventes en Europe')
plt.xlabel('Ventes en millions')
plt.ylabel('Fréquence')
plt.show()

# Distribution des ventes au Japon
sns.histplot(games_sales_df['JP_Sales'], bins=30, kde=True)
plt.title('Distribution des ventes au Japon')
plt.xlabel('Ventes en millions')
plt.ylabel('Fréquence')
plt.show()

# Distribution des ventes mondiales
sns.histplot(games_sales_df['Global_Sales'], bins=30, kde=True)
plt.title('Distribution des ventes mondiales')
plt.xlabel('Ventes en millions')
plt.ylabel('Fréquence')
plt.show()

# 2.2 Répartition des plateformes

# Répartition des plateformes de jeux
platform_counts = games_sales_df['Platform'].value_counts()
sns.barplot(x=platform_counts.index, y=platform_counts.values)
plt.title('Répartition des plateformes de jeux')
plt.xlabel('Plateforme')
plt.ylabel('Nombre de jeux')
plt.xticks(rotation=90)
plt.show()

# Ventes mondiales par plateforme
platform_sales = games_sales_df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)
sns.barplot(x=platform_sales.index, y=platform_sales.values)
plt.title('Ventes mondiales par plateforme')
plt.xlabel('Plateforme')
plt.ylabel('Ventes en millions')
plt.xticks(rotation=90)
plt.show()

# 2.3 Répartition des genres

# Répartition des genres de jeux
genre_counts = games_sales_df['Genre'].value_counts()
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Répartition des genres de jeux')
plt.xlabel('Genre')
plt.ylabel('Nombre de jeux')
plt.xticks(rotation=90)
plt.show()

# Ventes mondiales par genre
genre_sales = games_sales_df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
sns.barplot(x=genre_sales.index, y=genre_sales.values)
plt.title('Ventes mondiales par genre')
plt.xlabel('Genre')
plt.ylabel('Ventes en millions')
plt.xticks(rotation=90)
plt.show()

# 2.4 Répartition des années de sorties

# Répartition des années de sortie
year_counts = games_sales_df['Year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values)
plt.title('Répartition des années de sortie')
plt.xlabel('Année de sortie')
plt.ylabel('Nombre de jeux')
plt.xticks(rotation=90)
plt.show()

# Ventes mondiales par année de sortie
year_sales = games_sales_df.groupby('Year')['Global_Sales'].sum().sort_values(ascending=False)
sns.lineplot(x=year_sales.index, y=year_sales.values)
plt.title('Ventes mondiales par année de sortie')
plt.xlabel('Année de sortie')
plt.ylabel('Ventes en millions')
plt.xticks(rotation=90)
plt.show()

# Nettoyage des données

# Vérifier les valeurs manquantes
print(games_sales_df.isnull().sum())
print(meta_games_df.isnull().sum())
print(banned_games_df.isnull().sum())
print(popular_games_df.isnull().sum())

# Supprimer les valeurs manquantes
games_sales_df = games_sales_df.dropna()
meta_games_df = meta_games_df.dropna()
banned_games_df = banned_games_df.dropna()
popular_games_df = popular_games_df.dropna()

# Vérifier les doublons
print(games_sales_df.duplicated().sum())
print(meta_games_df.duplicated().sum())
print(banned_games_df.duplicated().sum())
print(popular_games_df.duplicated().sum())

# Supprimer les doublons
games_sales_df = games_sales_df.drop_duplicates()
meta_games_df = meta_games_df.drop_duplicates()
banned_games_df = banned_games_df.drop_duplicates()
popular_games_df = popular_games_df.drop_duplicates()

# Intégration et préparation des données

# Nettoyage et normalisation des colonnes de fusion dans chaque DataFrame
games_sales_df['Name'] = games_sales_df['Name'].str.strip()
games_sales_df['Platform'] = games_sales_df['Platform'].str.strip()

meta_games_df['name'] = meta_games_df['name'].str.strip()
meta_games_df['platform'] = meta_games_df['platform'].str.strip()

banned_games_df['Game'] = banned_games_df['Game'].str.strip()
banned_games_df['Country'] = banned_games_df['Country'].str.strip()

# Suppression des colonnes non nécessaires pour la fusion
banned_games_df = banned_games_df[['Game', 'Country']]

# Fusion des DataFrames
merged_data = pd.merge(games_sales_df, meta_games_df, left_on=['Name', 'Platform'], right_on=['name', 'platform'], how='inner')

# Remplacer 'tbd' par NaN dans la colonne user_review
merged_data['user_review'] = pd.to_numeric(merged_data['user_review'], errors='coerce')

merged_data['Name'] = merged_data['Name'].str.strip().str.lower()
banned_games_df['Game'] = banned_games_df['Game'].str.strip().str.lower()

# Fusionner les données en ajoutant une colonne pour marquer les jeux bannis
merged_data = merged_data.merge(banned_games_df, how='left', left_on='Name', right_on='Game', indicator=True)
merged_data['Banned'] = merged_data['_merge'] == 'both'

# Suppression des colonnes inutiles
merged_data.drop(columns=['Game', '_merge'], inplace=True)

# Affichage des informations du DataFrame combiné
print(merged_data.info())
print(merged_data.head())


# Fusionner games_sales_df avec popular_games_df
merged_data_popular = pd.merge(games_sales_df, popular_games_df, left_on='Name', right_on='Title', how='left')

# Afficher les informations du Dataframe combiné
print(merged_data_popular.info())

# Analyse des données

# Analyse exploratoire des données
sns.scatterplot(data=merged_data, x='meta_score', y='Global_Sales')
sns.scatterplot(data=merged_data, x='user_review', y='Global_Sales')
plt.show()

# Analyse de l'impact des scores sur les ventes
correlation_meta = merged_data['meta_score'].corr(merged_data['Global_Sales'])
correlation_user = merged_data['user_review'].corr(merged_data['Global_Sales'])

print(f"Correlation entre meta_score et Global_Sales: {correlation_meta}")
print(f"Correlation entre user_review et Global_Sales: {correlation_user}")

# Calculer les corrélations entre les ventes et les scores
correlation_matrix = merged_data[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'meta_score', 'user_review']].corr()
print(correlation_matrix)

# Visualiser les corrélations
sns.heatmap(correlation_matrix, annot=True)
plt.title('Corrélation entre les ventes et les scores des jeux')
plt.show()

# Analyse des ventes de jeux vidéo

# Analyse des Ventes Par Région
top_10_games = merged_data.nlargest(10, 'Global_Sales')[['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

# Plot des ventes par région pour les 10 jeux les plus vendus
plt.figure(figsize=(14, 8))
top_10_games.set_index('Name')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].plot(kind='bar', stacked=True)
plt.title('Top 10 Jeux les Plus Vendus')
plt.xlabel('Jeux')
plt.ylabel('Ventes (en millions)')
plt.legend(title='Région')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calcul des ventes totales par région
region_sales = games_sales_df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
print(region_sales)

# Visualisation des ventes totales par région
region_sales.plot(kind='bar')
plt.title('Ventes totales de jeux vidéo par région')
plt.ylabel('Ventes (en millions)')
plt.xlabel('Région')
plt.show()

# Analyse des ventes par Genre et Région

# Ventes par genre et par région
genre_region_sales = games_sales_df.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
print(genre_region_sales)

# Visualisation des ventes par genre et par région
genre_region_sales.plot(kind='bar', stacked=True)
plt.title('Ventes de jeux vidéo par genre et par région')
plt.ylabel('Ventes (en millions)')
plt.xlabel('Genre')
plt.show()

# Distribution des genres par région
for region in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']:
    genre_distribution = games_sales_df.groupby('Genre')[region].sum()
    genre_distribution.plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Distribution des genres de jeux vidéo en {region}')
    plt.ylabel('')
    plt.show()


# Analyse des Ventes Par Plateforme et Région

# Calcul des ventes par plateforme et par région
platform_region_sales = merged_data.groupby('Platform')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()

# Tri des ventes par plateforme (ordre décroissant)
platform_region_sales_triee = platform_region_sales.sort_values(by='NA_Sales', ascending=False)

# Top 10 des plateformes
top_10_platforms = platform_region_sales_triee.head(10)

# Création du graphique à barres
top_10_platforms.plot(kind="bar", stacked=False)
plt.xlabel("Plateforme")
plt.ylabel("Ventes totales")
plt.title("Top des plateformes de jeux vidéo")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Visualisation des ventes par plateforme et par région
platform_region_sales.plot(kind='bar', stacked=True)
plt.title('Ventes de jeux vidéo par plateforme et par région')
plt.ylabel('Ventes (en millions)')
plt.xlabel('Plateforme')
plt.show()

# Distribution des plateformes par région
for region in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']:
    platform_distribution = games_sales_df.groupby('Platform')[region].sum()
    platform_distribution.plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Distribution des plateformes de jeux vidéo en {region}')
    plt.ylabel('')
    plt.show()

# Analyse des ventes par régions restreintes

# Filtrer les jeux bannis
banned_games = merged_data[merged_data['Banned'] == True]

# Compter le nombre de jeux bannis par genre
banned_genre_counts = banned_games['Genre'].value_counts()

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(banned_genre_counts.index, banned_genre_counts.values, color='salmon')

plt.xlabel('Genre')
plt.ylabel('Nombre de Jeux Bannis')
plt.title('Nombre de Jeux Bannis par Genre')

# Ajout des valeurs sur les barres
for bar, count in zip(bars, banned_genre_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{count}', ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calcul des ventes par région pour les jeux bannis et non bannis
region_sales = merged_data.groupby('Banned')[['NA_Sales', 'EU_Sales', 'JP_Sales']].mean().transpose()

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x=region_sales.index, y=region_sales[False], color='b', alpha=0.7, label='Non Bannis')
sns.barplot(x=region_sales.index, y=region_sales[True], color='r', alpha=0.7, label='Bannis')
plt.title('Comparaison des Ventes Moyennes par Région pour les Jeux Bannis et Non Bannis')
plt.xlabel('Région de Ventes')
plt.ylabel('Ventes Moyennes (en millions)')
plt.legend()
plt.show()

# Calcul des ventes totales par plateforme pour les jeux bannis et non bannis
platform_sales = merged_data.groupby(['Platform', 'Banned'])['Global_Sales'].sum().unstack()

# Plot
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
platform_sales[False].plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
plt.title('Répartition des Ventes par Plateforme (Non Bannis)')

plt.subplot(1, 2, 2)
platform_sales[True].plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightsalmon'])
plt.title('Répartition des Ventes par Plateforme (Bannis)')

plt.tight_layout()
plt.show()

# Analyse de l'impact de la popularité des jeux sur les ventes

# Comparaison des ventes globales par type de jeu (populaire vs non populaire)
total_sales_popular = merged_data_popular[merged_data_popular['Title'].notnull()]['Global_Sales'].sum()
total_sales_non_popular = merged_data_popular[merged_data_popular['Title'].isnull()]['Global_Sales'].sum()

print(f"Ventes totales des jeux populaires : {total_sales_popular} millions")
print(f"Ventes totales des jeux non populaires : {total_sales_non_popular} millions")

# Marquer les jeux populaires
merged_data_popular['Popular'] = merged_data_popular['Title'].notnull()

# Comparaison des ventes globales par type de jeu (populaire vs non populaire)
plt.figure(figsize=(10, 6))
sns.barplot(x='Popular', y='Global_Sales', data=merged_data_popular, estimator=sum, ci=None, palette=['blue', 'red'])
plt.title('Impact des jeux populaires sur les ventes globales')
plt.xlabel('Jeu Populaire')
plt.ylabel('Ventes globales (en millions)')
plt.xticks([0, 1], ['Non Populaire', 'Populaire'])
plt.show()

# Calculer les ventes totales par région pour les jeux populaires et non populaires
sales_by_region = merged_data_popular.groupby('Popular').agg({
    'NA_Sales': 'sum',
    'EU_Sales': 'sum',
    'JP_Sales': 'sum',
    'Other_Sales': 'sum'
}).reset_index()

# Transformer le DataFrame pour un graphique à barres empilées
sales_by_region_melted = sales_by_region.melt(id_vars='Popular', var_name='Region', value_name='Sales')

# Graphique à barres empilées
plt.figure(figsize=(12, 8))
sns.barplot(x='Popular', y='Sales', hue='Region', data=sales_by_region_melted, palette='Set2')
plt.title('Répartition des ventes par région pour les jeux populaires et non populaires')
plt.xlabel('Jeu Populaire')
plt.ylabel('Ventes (en millions)')
plt.xticks([0, 1], ['Non Populaire', 'Populaire'])
plt.legend(title='Région')
plt.show()
