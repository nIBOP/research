import os
import pandas as pd
import zipfile
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import torch
import subprocess

# --- АУТЕНТИФИКАЦИЯ KAGGLE ---
# Используем предоставленный токен
os.environ['KAGGLE_USERNAME'] = "nibopov"
os.environ['KAGGLE_KEY'] = "2441e6151493c4f695ecea6a87bbd94b"
# -----------------------------

print("📥 Скачиваем данные с Kaggle (The Movies Dataset)...")

# Проверяем наличие распакованного файла
if not os.path.exists('movies_metadata.csv'):
    # Если архива нет, скачиваем
    if not os.path.exists('the-movies-dataset.zip'):
        print("⏳ Начало загрузки (около 200-900 МБ). Не прерывайте процесс!")
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "rounakbanik/the-movies-dataset"
        ], check=True)

    # Если архив есть (скачан только что или ранее), распаковываем
    if os.path.exists('the-movies-dataset.zip'):
        print("📦 Распаковка архива...")
        with zipfile.ZipFile('the-movies-dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    else:
        print("❌ Не удалось скачать архив. Проверьте соединение.")
else:
    print("✅ Данные уже скачаны и распакованы.")

print("⚙️ Обрабатываем датасет...")

if os.path.exists('movies_metadata.csv') and os.path.exists('ratings.csv'):
    # Читаем метаданные фильмов
    df_movies = pd.read_csv('movies_metadata.csv', usecols=['id', 'title', 'overview'], low_memory=False)

    # Очистка данных
    df_movies = df_movies[df_movies['id'].str.isnumeric() == True]
    df_movies['id'] = df_movies['id'].astype(int)
    # --- FIX: Убираем дубликаты ID ---
    df_movies = df_movies.drop_duplicates(subset=['id'])
    df_movies = df_movies.dropna(subset=['overview', 'title']) # Убираем пустые

    # Читаем рейтинги
    df_ratings = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'])
    df_ratings_1m = df_ratings.head(1000000) # Берем первый 1М взаимодействий

    # Синхронизация ID
    valid_movie_ids = set(df_movies['id']).intersection(set(df_ratings_1m['movieId']))

    df_movies_final = df_movies[df_movies['id'].isin(valid_movie_ids)].reset_index(drop=True)
    df_ratings_final = df_ratings_1m[df_ratings_1m['movieId'].isin(valid_movie_ids)].reset_index(drop=True)

    # Переименовываем для удобства
    df_movies_final = df_movies_final.rename(columns={'overview': 'description', 'id': 'movie_id'})
    df_ratings_final = df_ratings_final.rename(columns={'movieId': 'movie_id', 'userId': 'user_id'})

    print(f"✅ Готово! Сформированы датафреймы.")
    print(f"   Фильмов с описанием: {len(df_movies_final)}")
    print(f"   Рейтингов: {len(df_ratings_final)}")
    print(df_movies_final[['movie_id', 'title', 'description']].head(3))
else:
    raise FileNotFoundError("❌ Ошибка: Файлы данных (movies_metadata.csv или ratings.csv) не найдены. Пожалуйста, проверьте наличие файлов!")

print("🧠 Векторизация текстов (Sentence-BERT)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
# Получаем эмбеддинги для всех фильмов
embeddings = model.encode(df_movies_final['description'].tolist(), show_progress_bar=True)

print("📊 Кластеризация K-Means (150 кластеров)...")
kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
kmeans.fit(embeddings)

os.makedirs('clean_movies', exist_ok=True)

# 1. Сохраняем центроиды (для адаптивной модели)
centroids_tensor = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
torch.save(centroids_tensor, 'cluster_centroids.pt')

# 2. Создаем файл свойств (.item) для RecBole
df_item = pd.DataFrame({
    'item_id:token': df_movies_final['movie_id'],
    'cluster_id:token': kmeans.labels_
})
df_item.to_csv('clean_movies/clean_movies.item', sep='\t', index=False)

# 3. Создаем чистый файл взаимодействий (.inter)
df_inter = df_ratings_final[['user_id', 'movie_id', 'rating']].copy()
df_inter.columns = ['user_id:token', 'item_id:token', 'rating:float']
df_inter.to_csv('clean_movies/clean_movies.inter', index=False, sep='\t')

print("✅ Датасет clean_movies и центроиды успешно сформированы!")