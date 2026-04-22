# Book Recommendation System

This repository contains a simple, modular Book Recommendation System built as a solution for the take-home assignment.

## Methodology & Framing

The main objective is to **predict what a user should read next**. After exploring the dimensions of the dataset, I decided to scope the problem around **recommending the next new book to start**, rather than next chapters in a currently-read book.

Since interactions specify that a user read a *chapter* rather than a full book, we get sequential signals. Reading many chapters of a given book is a strong positive signal, whereas dropping off after Chapter 1 is a weak or negative signal.

### Assumptions & Implementation
1. **Implicit Ratings via Completion Rates:** 
   I aggregated the dataset so that for each `(user_id, book_id)` pair, we compute a `completion_rate`. `completion_rate = chapters_read / total_chapters_in_book`. This allowed me to treat the recommendation system as an implicit rating problem, heavily weighting books that a user finished.
2. **Algorithm:**
   Due to time constraints and the sparsity of reading histories, I opted for a **Content-Based Filtering** approach. I represented each book as a TF-IDF vector of its combined `tags` and `author_id`.
3. **User Profiling:**
   A user's preferences are modeled by computing a weighted average of the TF-IDF vectors of the books they have interacted with, where the weights correspond to their `completion_rate`.
4. **Cold Start & Popularity Hybridization:**
   For users with no reading history, the system falls back to recommending **globally popular books** (books with the highest sum of completion rates across all users). Even for users with a history, I injected a minor popularity bias to the similarity scores to break ties and favour highly completed global literature.
5. **Lack of Chronological Timestamps:**
   `interactions.csv` does not contain explicit timestamps, meaning I couldn't evaluate using a strict chronologically-split Next-Item prediction. Instead, I evaluated via a *Leave-One-Out (Highly Completed)* hold-out method.

## Tradeoffs Made
1. **Content-Based vs Collaborative Filtering:** With more time and computational allowance, an implicit Matrix Factorization technique (like BPR using the `implicit` library) or a two-tower Neural Collaborative Filtering model might yield better serendipity. I chose Content-Based TF-IDF as it is robust, easy to reason about, and has light external dependencies (only `scikit-learn` and `pandas`).
2. **Ignoring Sequential Progression Across Multiple Books:** If users typically hop between different series, a sequence modeling approach (e.g. BERT4Rec, GRU4Rec) could capture short-term and long-term intents better.
3. **Truncating Memory Matrix Considerations:** Currently, similarity is computed via in-memory sparse matrix multiplications which is scalable locally up to hundreds of thousands of books, but would need an approximate nearest neighbor system (like FAISS) if it were millions of items.

## How to Run

### Install Dependencies
Ensure you have `pandas`, `scikit-learn`, and `numpy` installed:
```bash
pip install -r requirements.txt
```

### Get Recommendations for a Single User
You can supply a specific `user_id` to get top 5 book recommendations:
```bash
python recommender.py --user user_2378720
```

### Run Offline Evaluation
You can also run an offline evaluation on a sampled hold-out user set:
```bash
python recommender.py --eval
```
*Note: Make sure both `chapters.csv` and `interactions.csv` are in the same directory as the script.*
