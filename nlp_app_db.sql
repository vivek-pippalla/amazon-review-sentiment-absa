CREATE DATABASE nlp_app_db;
USE nlp_app_db;

CREATE TABLE reviews(
    id INT AUTO_INCREMENT PRIMARY KEY,
    review_text TEXT NOT NULL,
    summarized_review TEXT,
    sentiment VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(review_text(255))
);

CREATE TABLE absa_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    review_id INT NOT NULL,
    aspect VARCHAR(255) NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE
);


SELECT * FROM reviews;