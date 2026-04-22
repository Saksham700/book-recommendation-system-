import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys

class BookRecommender:
    def __init__(self, chapters_path='chapters.csv', interactions_path='interactions.csv'):
        self.chapters_path = chapters_path
        self.interactions_path = interactions_path
        self.df_c = None
        self.df_i = None
        self.books = None
        self.user_book = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.popular_books = []
        
    def load_and_preprocess(self):
        print("Loading data...")
        self.df_c = pd.read_csv(self.chapters_path)
        self.df_i = pd.read_csv(self.interactions_path)
        
        print("Preprocessing tags and authors...")
        self.df_c['tags'] = self.df_c['tags'].fillna('')
        
        # Total chapters per book
        book_chapters_count = self.df_c.groupby('book_id').size().reset_index(name='total_chapters')
        
        # Unique tags and author
        self.books = self.df_c.groupby('book_id').agg({
            'author_id': 'first',
            'tags': lambda x: '|'.join(set('|'.join(x).split('|'))),
        }).reset_index()
        
        self.books = pd.merge(self.books, book_chapters_count, on='book_id')
        
        print("Processing user interactions for implicit signals...")
        # A user's total active chapters in a book
        user_book = self.df_i.groupby(['user_id', 'book_id']).size().reset_index(name='chapters_read')
        
        # Merge to find completion rate
        user_book = pd.merge(user_book, self.books[['book_id', 'total_chapters']], on='book_id')
        user_book['completion_rate'] = user_book['chapters_read'] / user_book['total_chapters']
        user_book['completion_rate'] = user_book['completion_rate'].clip(upper=1.0)
        self.user_book = user_book
        
        print("Computing book popularity profiles...")
        # Popularity scoring for cold start
        pop = self.user_book.groupby('book_id')['completion_rate'].sum().reset_index(name='pop_score')
        self.books = pd.merge(self.books, pop, on='book_id', how='left').fillna({'pop_score': 0})
        
        # Sort books by popularity descending (this fixes their indices)
        self.books = self.books.sort_values('pop_score', ascending=False).reset_index(drop=True)
        self.popular_books = self.books['book_id'].tolist()
        
        # Lookup structures
        self.book_to_idx = pd.Series(self.books.index, index=self.books['book_id'].values).to_dict()
        
        print("Building vector spaces...")
        # TF-IDF Features: Author + all unique tags
        self.books['content_features'] = self.books['tags'].str.replace('|', ' ', regex=False) + ' author_' + self.books['author_id'].astype(str)
        
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.books['content_features'])
        
    def recommend(self, user_id, top_n=5):
        if user_id not in self.user_book['user_id'].values:
            return self.popular_books[:top_n]
            
        user_history = self.user_book[self.user_book['user_id'] == user_id]
        user_books = user_history['book_id'].tolist()
        user_weights = user_history['completion_rate'].values
        
        book_indices = [self.book_to_idx[b] for b in user_books if b in self.book_to_idx]
        if not book_indices:
            return self.popular_books[:top_n]
            
        user_book_vectors = self.tfidf_matrix[book_indices].toarray()
        user_weights = user_weights[:len(book_indices)]
        
        if user_weights.sum() > 0:
            user_weights = user_weights / user_weights.sum()
        else:
            user_weights = np.ones(len(user_weights)) / len(user_weights)
            
        # Compute user's average content taste
        user_profile = np.average(user_book_vectors, axis=0, weights=user_weights)
        
        # Compute similarities across all books
        sims = cosine_similarity([user_profile], self.tfidf_matrix)[0]
        
        # Sort scores natively
        # Add a very minor popularity boost to break ties
        pop_scores = self.books['pop_score'].values
        max_pop = pop_scores.max() if pop_scores.max() > 0 else 1
        
        combined_scores = sims + 0.05 * (pop_scores / max_pop)
        
        # Get top indices (we get a bit more than top_n to account for filtering read books)
        top_indices = combined_scores.argsort()[::-1]
        
        recommendations = []
        for idx in top_indices:
            b_id = self.books.iloc[idx]['book_id']
            if b_id not in user_books:
                recommendations.append(b_id)
            if len(recommendations) >= top_n:
                break
                
        return recommendations

    def evaluate(self):
        print("Running offline evaluation on a sampled hold-out set...")
        sample_users = self.user_book['user_id'].drop_duplicates().sample(1000, random_state=42).tolist()
        hits = 0
        valid_evals = 0
        
        for u in sample_users:
            u_history = self.user_book[self.user_book['user_id'] == u].copy()
            # Only evaluate users with at least 3 interacted books to ensure they have enough history
            if len(u_history) < 3:
                continue
                
            # Hide the most "completed" book as the target
            u_history = u_history.sort_values('completion_rate', ascending=False)
            target_book = u_history.iloc[0]['book_id']
            
            train_books = set(u_history.iloc[1:]['book_id'].tolist())
            train_weights = u_history.iloc[1:]['completion_rate'].values
            
            book_indices = [self.book_to_idx[b] for b in train_books if b in self.book_to_idx]
            if not book_indices:
                continue
                
            user_book_vectors = self.tfidf_matrix[book_indices].toarray()
            
            if train_weights.sum() > 0:
                train_weights = train_weights / train_weights.sum()
            else:
                train_weights = np.ones(len(train_weights)) / len(train_weights)
                
            user_profile = np.average(user_book_vectors, axis=0, weights=train_weights)
            sims = cosine_similarity([user_profile], self.tfidf_matrix)[0]
            
            pop_scores = self.books['pop_score'].values
            max_pop = pop_scores.max() if pop_scores.max() > 0 else 1
            combined_scores = sims + 0.05 * (pop_scores / max_pop)
            
            top_indices = combined_scores.argsort()[::-1]
            
            top_10 = []
            for idx in top_indices:
                b_id = self.books.iloc[idx]['book_id']
                if b_id not in train_books:
                    top_10.append(b_id)
                if len(top_10) >= 10:
                    break
                    
            if target_book in top_10:
                hits += 1
            valid_evals += 1
            
        print(f"Users Evaluated (>= 3 books history): {valid_evals}")
        if valid_evals > 0:
            print(f"Recall@10: {hits / valid_evals:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Book Recommendation System")
    parser.add_argument('--user', type=str, help='User ID to recommend books for')
    parser.add_argument('--eval', action='store_true', help='Run offline evaluation')
    args = parser.parse_args()
    
    if not args.eval and not args.user:
        print("Please provide --eval or --user <user_id>")
        sys.exit(1)
        
    rec = BookRecommender()
    rec.load_and_preprocess()
    
    if args.eval:
        rec.evaluate()
        
    if args.user:
        recs = rec.recommend(args.user)
        print(f"Top recommendations for {args.user}: {recs}")
