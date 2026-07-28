import pandas as pd
import ast
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px


#load dataset
credits=pd.read_csv("tmdb_5000_credits.csv")
movies=pd.read_csv("tmdb_5000_movies.csv")

#merge movie ad credits data
movies=movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)

#select columns
movies=movies[
    [
    "title_x",
    "genres",
    "keywords",
    "cast",
    "crew",
    "overview",
    "vote_average",
    "vote_count"
    ]
]

#Rename title_x to title
movies.rename(
    columns={
        "title_x":"title"
    },
    inplace=True
)

#handle missing values
movies["overview"]=movies["overview"].fillna("")


def convert(obj):
    result=[]

    try: 
           
        data=ast.literal_eval(obj)

        for item in data:

            result.append(
                item["name"]

            )
        return result

    except:

        return[]


#apply functions
movies["genres"]=movies["genres"].apply(convert)

movies["keywords"]=(
movies["keywords"].apply(convert)
)

#extract top 3 cast members

def convert_cast(obj):
    result=[]

    try:

        data=ast.literal_eval(obj)
        for item in data[:3]:

            result.append(
                item["name"]
            )
        return result
    except:

        return[]

movies["cast"]=movies["cast"].apply(convert_cast)

#extract director

def get_director(obj):

    result=[]

    try:

        data=ast.literal_eval(obj)
        for item in data:

            if item["job"]=="Director":
                result.append(
                    item["name"]
                )

        return result
    except:
            return[]
movies["crew"]=movies["crew"].apply(get_director)


movies["overview"]=movies["overview"].apply(
    lambda x:x.split()
)
movies["genres"] = movies["genres"].apply(
lambda x: [
    i.replace(" ", "")
    for i in x
]
)


movies["keywords"] = movies["keywords"].apply(
lambda x: [
    i.replace(" ", "")
    for i in x
]
)


movies["cast"] = movies["cast"].apply(
lambda x: [
    i.replace(" ", "")
    for i in x
]
)


movies["crew"] = movies["crew"].apply(
lambda x: [
    i.replace(" ", "")
    for i in x
]
)

#create tags columns
movies["tags"]=(
    movies["overview"]
    +movies["genres"]
    +movies["keywords"]
    +movies["cast"]
    +movies["crew"]

)

#final dataframe
new_df=movies[
    [
    "title",
    "tags",
    "vote_average",
    "vote_count"
 ]
].copy()

#convert list into string
new_df["tags"]=new_df["tags"].apply(
    lambda x:" ".join(x)
)

new_df["tags"]=new_df["tags"].apply(
    lambda x:x.lower()
)

cv = CountVectorizer(
max_features=5000,
stop_words="english"
)


vectors = cv.fit_transform(
new_df["tags"]
).toarray()


similarity=cosine_similarity(
vectors
)  
def recommend(movie_name):

    # Find movie index

    movie_index = new_df[
        new_df["title"].str.lower()
        == movie_name.lower()
    ].index


    # If movie does not exist

    if len(movie_index) == 0:

        return None


    index = movie_index[0]


    # Get similarity scores

    distances = similarity[index]


    # Sort movies based on similarity

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]


    recommendations = []


    # Store top 5 movies

    for i, score in movie_list:

        recommendations.append(
            {
                "title": new_df.iloc[i]["title"],
                "similarity": round(
                    score * 100,
                    2
                ),
                "rating": new_df.iloc[i][
                    "vote_average"
                ],
                "votes": new_df.iloc[i][
                    "vote_count"
                ]
            }
        )


    return recommendations

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

#  TITLE


st.title(
    "🎬 Content-Based Movie Recommendation Engine"
)


st.write(
    "Enter the name of a movie and get "
    "5 similar movie recommendations."
)



#  MOVIE SEARCH BOX


movie_name = st.text_input(
    "Enter your favorite movie",
    placeholder="Example: Avatar"
)



# RECOMMENDATION BUTTON


if st.button("Recommend Movies"):


    # Check empty input

    if movie_name.strip() == "":

        st.warning(
            "Please enter a movie name."
        )


    else:

        # Get recommendations

        recommendations = recommend(movie_name)

        # 19. IF MOVIE NOT FOUND


        if recommendations is None:

            st.error(
                "Movie not found. "
                "Please enter the exact movie name."
            )   


        #  DISPLAY RECOMMENDATIONS
        

        else:

            st.subheader(
                "Top 5 Similar Movies"
            )


            # Create 5 columns

            columns = st.columns(5)


            # Display each movie

            for column, movie in zip(
                columns,
                recommendations
            ):

                with column:

                    st.markdown(
                        f"### {movie['title']}"
                    )


                    st.metric(
                        "Similarity",
                        f"{movie['similarity']}%"
                    )


                    st.write(
                        f"⭐ Rating: "
                        f"{movie['rating']}"
                    )


                    st.write(
                        f"👥 Votes: "
                        f"{movie['votes']}"
                    )


            
            #  CREATE DATAFRAME FOR PLOTLY
            

            chart_data = pd.DataFrame(
                recommendations
            )


            
            #  CREATE PLOTLY BAR CHART
            

            fig = px.bar(

                chart_data,

                x="similarity",

                y="title",

                orientation="h",

                title="Top 5 Similar Movies",

                labels={
                    "similarity":
                    "Similarity (%)",

                    "title":
                    "Movie"
                },

                text="similarity"
            )


            # Display percentage on bars

            fig.update_traces(

                texttemplate="%{text}%",

                textposition="outside"
            )


            # Arrange movies from highest to lowest

            fig.update_layout(

                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )


            # Display chart

            st.plotly_chart(
                fig,
                use_container_width=True
            )