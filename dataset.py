import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

path = '/Users/isabella/Desktop/dataset-ml-25m/'
movies = pd.read_csv(path + 'movies.csv', sep=',')

genres_list = ["Action","Adventure","Animation","Children",
"Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","IMAX",
"Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western","(no genres listed)"]

#one hot encoding genres movies
mlb = MultiLabelBinarizer(classes=genres_list)
movies_genres = movies['genres'].str.get_dummies(sep='|')
movies_genres = movies_genres[genres_list]
movies = pd.concat([movies, movies_genres], axis=1)
movies = movies.drop('genres', axis=1)

# Merge tag and movies and create columns for each tag
gen_tags = pd.read_csv(path + 'genome-tags.csv', sep=',')
gen_scores = pd.read_csv(path + 'genome-scores.csv', sep=',')

df = movies.merge(gen_scores,on='movieId')
df = df.merge(gen_tags, on='tagId')
df =df.pivot_table(index=['movieId','title','Action','Adventure','Animation','Children','Comedy','Crime',
                'Documentary','Drama','Fantasy','Film-Noir','Horror',
                'IMAX','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western','(no genres listed)'],
                columns='tag', values='relevance', fill_value=0).reset_index().rename_axis(None,axis=1)

#Merge movies_genome with film rating
ratings = pd.read_csv(path + 'ratings.csv', sep=',')
movies = pd.read_csv(path + 'movies_genome.csv', sep=',')

ratings = ratings.groupby(['movieId'])['rating'].mean().reset_index()

#Rounded to the nearest 0.5
def myround(x, prec=2, base=.5):
  return round(base * round(float(x)/base),prec)

ratings["rating"] = ratings["rating"].apply(lambda x:myround(x))
df = ratings.merge(movies,on='movieId')
dataset = df.drop('title', axis=1)

dataset.to_csv(path + "dataset.csv",index=False)



