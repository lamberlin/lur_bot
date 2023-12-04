import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import numpy as np
from collections import Counter
import pickle
import zipfile
import os
class allEmbeddingsCalculator:
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open('all_stop_words.pkl', 'rb') as file:
            self.stop_words = pickle.load(file)
        self.embedding_file='reviews_embedding_bert.zip'

    def preprocess(self, text):
        words = self.tokenizer.tokenize(text.lower())
        return [w for w in words if w not in self.stop_words]

    def text_to_bert_embedding(self, text):
        self.model = self.model.to(self.device)
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            input_ids = torch.tensor(tokens).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
            # embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            embedding = torch.mean(outputs[0], dim=1).cpu().numpy()
            return embedding
        except:
            return [[]]
    def load_reviews(self, zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("extracted_data")
            npy_file = [f for f in os.listdir("extracted_data") if f.endswith('.npy')][0]
            npy_file_path = os.path.join("extracted_data", npy_file)
            data = np.load(npy_file_path, allow_pickle=True).item()
            os.remove(npy_file_path)
            os.rmdir("extracted_data")
        return data['embeddings'], data['reviews'], data['University'], data['sub_topic'], data['author'], data['created']


    def calculate_topic_probabilities_with_reviews(self, input_text):
        embeddings, reviews, universities, sub_topics, authors, created_dates = self.load_reviews(self.embedding_file)
        input_text_tokens = self.preprocess(input_text)
        input_text_embedding = self.text_to_bert_embedding(" ".join(input_text_tokens))
        similarities = cosine_similarity(input_text_embedding.reshape(1, -1), embeddings).flatten()
        topic_scores = {}
        for score, topic in zip(similarities, sub_topics):
            topic_scores[topic] = topic_scores.get(topic, 0) + score
        top_topics = sorted(topic_scores, key=topic_scores.get, reverse=True)[:3]
        total_topic_score = sum(topic_scores[topic] for topic in top_topics)
        topic_probabilities = {topic: topic_scores[topic] / total_topic_score for topic in top_topics}
        filtered_indices = [i for i, topic in enumerate(sub_topics) if topic in top_topics]
        filtered_data = {
            'sub_topic': [sub_topics[i] for i in filtered_indices],
            'review': [reviews[i] for i in filtered_indices],
            'similarity_score': [similarities[i] for i in filtered_indices],
            'University': [universities[i] for i in filtered_indices],
            'author': [authors[i] for i in filtered_indices],
            'created': [created_dates[i] for i in filtered_indices],
            'topic_probability': [topic_probabilities[sub_topics[i]] for i in filtered_indices]
        }
        filtered_df = pd.DataFrame(filtered_data)
        return filtered_df
 

    def calculate_weighted_average_similarity(self, input_text):
        filtered_df = self.calculate_topic_probabilities_with_reviews(input_text)

        filtered_df['University'] = filtered_df['University'].astype(str)
        filtered_df['sub_topic'] = filtered_df['sub_topic'].astype(str)

        mean_scores = filtered_df.groupby(['University', 'sub_topic']).agg({'similarity_score': 'mean'}).reset_index()
        mean_scores['weighted_score'] = mean_scores.apply(
            lambda x: x['similarity_score'] * filtered_df[filtered_df['sub_topic'] == x['sub_topic']]['topic_probability'].iloc[0], 
            axis=1
        )
        was_scores = mean_scores.groupby('University')['weighted_score'].sum().reset_index().rename(columns={'weighted_score': 'WAS'})

        total_was = was_scores['WAS'].sum()
        was_scores['WAS'] = was_scores['WAS'] / total_was

        highest_prob_topic = mean_scores.loc[mean_scores.groupby('University')['similarity_score'].idxmax()][['University', 'sub_topic']]
        highest_prob_topic.columns = ['University', 'Highest_Prob_Topic']
        highest_prob_topic['University'] = highest_prob_topic['University'].astype(str)
        highest_prob_topic['Highest_Prob_Topic'] = highest_prob_topic['Highest_Prob_Topic'].astype(str)

        result = pd.merge(was_scores, highest_prob_topic, on='University')

        relevant_review = filtered_df.loc[filtered_df.groupby(['University', 'sub_topic'])['similarity_score'].idxmax()][['University', 'sub_topic', 'review', 'author', 'created']]
        relevant_review.columns = ['University', 'Highest_Prob_Topic', 'Most_Relevant_Review', 'Most_Relevant_Author', 'Most_Relevant_Created']
        relevant_review['University'] = relevant_review['University'].astype(str)
        relevant_review['Highest_Prob_Topic'] = relevant_review['Highest_Prob_Topic'].astype(str)

        final_result = pd.merge(result, relevant_review, on=['University', 'Highest_Prob_Topic'])

        return final_result[['University', 'WAS', 'Highest_Prob_Topic', 'Most_Relevant_Review', 'Most_Relevant_Author', 'Most_Relevant_Created']]
    
    def calculate_confidence_score(self, input_text):
        df = self.calculate_topic_probabilities_with_reviews(input_text)
        
        if df.empty:
            return 0

        topic_counts = df['sub_topic'].value_counts()
        total_data_points = len(df)
        topic_portions = topic_counts / total_data_points
        topic_probabilities = df['topic_probability'].groupby(df['sub_topic']).mean()
        weighted_average = sum(topic_portions * topic_probabilities)

        return weighted_average

