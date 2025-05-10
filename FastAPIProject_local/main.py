from fastapi import FastAPI,Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import pyodbc
import random
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import pandas as pd
import os
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

app = FastAPI()

# Global caches
big_data_cache = None
content_similarity_cache = None
recommdation2_cache = None

# ✅ Connection details
server = 'db17915.public.databaseasp.net'
database = 'db17915'
username = 'db17915'
password = 'wQ+5A!6n?zR7'

# ✅ Try drivers in order
preferred_drivers = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
    "SQL Server"
]

def get_working_driver():
    installed_drivers = pyodbc.drivers()
    for driver in preferred_drivers:
        if driver in installed_drivers:
            return driver
    raise Exception(f"No compatible ODBC drivers found. Please install one of: {preferred_drivers}")


def get_db_connection():
    try:
        driver = get_working_driver()
        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"Encrypt=no;"
        )
        return pyodbc.connect(connection_string)
    except Exception as conn_error:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {conn_error}")

# ✅ SQLAlchemy engine using working driver
try:
    driver = get_working_driver()
    connection_url = URL.create(
        "mssql+pyodbc",
        username=username,
        password=password,
        host=server,
        database=database,
        query={"driver": driver, "Encrypt": "no"}
    )
    engine = create_engine(connection_url)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"SQLAlchemy engine init failed: {e}")

# ✅ DTO model for response
class BookDto(BaseModel):
    id: str = Field(..., alias="id")
    name: str = Field(..., alias="name")
    author: str = Field(..., alias="author")
    description: str = Field(..., alias="description")
    file_url: str = Field(..., alias="fileURL")
    category_name: str = Field(..., alias="categoryName")
    published_year: int = Field(..., alias="publishedYear")
    average_rating: float = Field(..., alias="averageRating")
    num_pages: int = Field(..., alias="numPages")
    link_book: Optional[str] = Field(None, alias="linkBook")
    is_favorite: bool = Field(..., alias="isFavorite")

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True
        by_alias = True

def map_to_book_dto(row, favorite_book_ids) -> BookDto:
    book_id = str(row.get("book_id") or row.get("id", "")).strip()
    return BookDto(
        id=book_id,
        name=row.get("title") or row.get("name", ""),
        author=row.get("author", ""),
        description=row.get("description", ""),
        file_url=row.get("book_pic") or row.get("fileURL", ""),
        category_name=row.get("categories") or row.get("categoryName", ""),
        published_year=int(row.get("published_year") or row.get("publishedYear", 0)),
        average_rating=float(row.get("average_rate") or row.get("averageRating", 0.0)),
        num_pages=int(row.get("num_pages") or row.get("numPages", 0)),
        link_book=row.get("linkBook"),
        is_favorite=book_id in favorite_book_ids
    )




# ✅ Generate recommendations endpoint
@app.get("/generate_recommendations/")
async def generate_recommendations(user_id: str) -> dict:
    global big_data_cache, content_similarity_cache, recommdation2_cache

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path1 = os.path.join(BASE_DIR, "last_filter.csv")

        user_reviews_df = pd.read_sql(
            text("SELECT UserId AS user_id, BookId AS book_id, Rating AS rating FROM Reviews WHERE UserId = :user_id"),
            engine, params={"user_id": user_id}
        )
        all_reviews_df = pd.read_sql(
            "SELECT UserId AS user_id, BookId AS book_id, Rating AS rating FROM Reviews", engine
        )

        big_data = pd.read_csv(file_path1, delimiter=';', on_bad_lines='skip')
        big_data.columns = big_data.columns.str.lower()

        def check_columns(df, required, name):
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing column(s) {missing} in {name} data")

        check_columns(user_reviews_df, ['user_id', 'book_id', 'rating'], 'user_reviews')
        check_columns(all_reviews_df, ['user_id', 'book_id', 'rating'], 'reviews')
        check_columns(big_data, ['book_id'], 'last_filter')

        user_reviews_df["book_id"] = user_reviews_df["book_id"].astype(str)
        book_set = set(user_reviews_df["book_id"])

        overlap_users = {}
        for _, row in all_reviews_df.iterrows():
            user_id_r = str(row['user_id'])
            book_id = str(row['book_id'])
            if book_id in book_set:
                overlap_users[user_id_r] = overlap_users.get(user_id_r, 0) + 1

        filtered_users = {k for k, v in overlap_users.items() if v >= 1}

        # Build filtered interactions
        interactions_list = []
        for index, row in all_reviews_df.iterrows():
            user_id = str(row['user_id'])
            book_id = str(row['book_id'])
            rating = row['rating']
            if user_id in filtered_users:
                interactions_list.append([user_id, book_id, rating])

        # ✅ SAFETY CHECK
        if not interactions_list:
            return {
                "recommended_books": [],
                "message": "No similar users found. Try uploading more liked books or check if book IDs match the dataset."
            }

        interactions_df = pd.DataFrame(interactions_list, columns=["user_id", "book_id", "rating"])
        interactions_df = pd.concat([user_reviews_df[['user_id', 'book_id', 'rating']], interactions_df])

        interactions_df["user_index"] = interactions_df["user_id"].astype("category").cat.codes
        interactions_df["book_index"] = interactions_df["book_id"].astype("category").cat.codes

        ratings_mat_coo = coo_matrix((interactions_df["rating"], (interactions_df["user_index"], interactions_df["book_index"])))
        ratings_mat = ratings_mat_coo.tocsr()

        my_index = 0
        user_similarity = cosine_similarity(ratings_mat[my_index, :], ratings_mat).flatten()

        indices = np.argpartition(user_similarity, -20)[-20:]
        similar_users = interactions_df[interactions_df["user_index"].isin(indices)]
        similar_users = similar_users[similar_users["user_id"] != user_id]

        book_recs = similar_users.groupby("book_id").rating.agg(['count', 'mean'])
        book_recs = book_recs[(book_recs["mean"] >= 3.5) & (book_recs["count"] > 1)].reset_index()

        top_n = 10
        recommended_ids = book_recs.sort_values(by="mean", ascending=False)["book_id"].astype(str).head(top_n).tolist()

        big_data_cache = big_data.copy()

        recommdation2 = big_data_cache.copy()
        recommdation2['tags'] = (
            recommdation2['author'].fillna('') + ' ' +
            recommdation2['description'].fillna('') + ' ' +
            recommdation2['categories'].fillna('')
        )
        recommdation2.drop(columns=['description', 'categories', 'author', 'book_pic', 'published_year', 'average_rate', 'num_pages'], errors='ignore', inplace=True)

        cv = CountVectorizer(max_features=10000, stop_words='english')
        vector = cv.fit_transform(recommdation2['tags'].values.astype('U')).toarray()
        content_similarity = cosine_similarity(vector)

        recommdation2_cache = recommdation2
        content_similarity_cache = content_similarity

        recommended_books_data = big_data_cache[big_data_cache['book_id'].astype(str).isin(recommended_ids)]

        #recommended_books_list = recommended_books_data.to_dict(orient='records')

        with engine.connect() as conn:
            fav_result = conn.execute(
                text("SELECT BookId FROM Favorites WHERE UserId = :user_id"),
                {"user_id": user_id}
            )
            favorite_book_ids = {str(row[0]) for row in fav_result.fetchall()}


        recommended_books_list = [
            map_to_book_dto(row, favorite_book_ids)
            for _, row in recommended_books_data.iterrows()
        ]

       # for book in recommended_books_list:
        #    book["isFavorite"] = book["book_id"] in favorite_book_ids

        return {
            "recommended_books": jsonable_encoder(recommended_books_list, by_alias=True),
            "message": "Collaborative recommendations generated successfully."
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/recommend_by_book_id/")
def recommend_by_book_id(
    book_id: str = Query(..., description="Book ID for content-based recommendation"),
    user_id: str = Query(..., description="User ID to check favorite books"),
    top_n: int = 10
):
    global recommdation2_cache, content_similarity_cache, big_data_cache

    if recommdation2_cache is None or content_similarity_cache is None or big_data_cache is None:
        raise HTTPException(
            status_code=400,
            detail="Please call /generate_recommendations/ first to upload and process the data."
        )

    try:
        # Normalize book_id and dataset
        book_id = book_id.strip()
        recommdation2_cache["book_id"] = recommdation2_cache["book_id"].astype(str).str.strip()

        if book_id not in recommdation2_cache["book_id"].values:
            return {
                "recommended_books": [],
                "message": f"Book ID '{book_id}' not found in the dataset."
            }

        # Find the index of the book
        book_index = recommdation2_cache[recommdation2_cache["book_id"] == book_id].index[0]
        similarity_scores = list(enumerate(content_similarity_cache[book_index]))

        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_similar_indices = [i for i, _ in sorted_scores if i != book_index][:top_n]
        recommended_ids = recommdation2_cache.iloc[top_similar_indices]["book_id"].astype(str).tolist()

        # Get full book details
        recommended_books_data = big_data_cache[big_data_cache['book_id'].astype(str).isin(recommended_ids)]
        #recommended_books_list = recommended_books_data.to_dict(orient='records')

        # Fetch user's favorites
        with engine.connect() as conn:
            fav_result = conn.execute(
                text("SELECT BookId FROM Favorites WHERE UserId = :user_id"),
                {"user_id": user_id}
            )
            favorite_book_ids = {str(row[0]) for row in fav_result.fetchall()}

        recommended_books_list = [
            map_to_book_dto(row, favorite_book_ids)
            for _, row in recommended_books_data.iterrows()
        ]
        #for book in recommended_books_list:
        #    book["isFavorite"] = book["book_id"] in favorite_book_ids

        return {
            "recommended_books": jsonable_encoder(recommended_books_list),

        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during content-based recommendation: {str(e)}")





