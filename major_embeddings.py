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
class EmbeddingsCalculator:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        with open('stop_words.pkl', 'rb') as file:
            self.stop_words = pickle.load(file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_file='major_embedding_bert.zip'

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
            embedding = torch.mean(outputs[0], dim=1).cpu().numpy()
            return embedding
        except Exception as e:
            print(f"Error in text_to_bert_embedding: {e}")
            return [[]]
    def load_reviews(self, zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("extracted_data")
            npy_file = [f for f in os.listdir("extracted_data") if f.endswith('.npy')][0]
            npy_file_path = os.path.join("extracted_data", npy_file)
            data = np.load(npy_file_path, allow_pickle=True).item()
            os.remove(npy_file_path)
            os.rmdir("extracted_data")
        return data['embeddings'], data['reviews'], data['University'], data['Professor'], data['major_field'], data['course'], data['created']
   
    def calculate_topic_probabilities_with_reviews(self, input_text):
        embeddings, reviews, universities, professors, sub_topics, course, created_dates = self.load_reviews(self.embedding_file)
        input_text_tokens = self.preprocess(input_text)
        input_text_embedding = self.text_to_bert_embedding(" ".join(input_text_tokens))
        similarities = cosine_similarity(input_text_embedding.reshape(1, -1), embeddings).flatten()
        topic_scores = {}
        topic_counts = {}

        for score, topic in zip(similarities, sub_topics):
            topic_scores[topic] = topic_scores.get(topic, 0) + score
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        for topic in topic_scores:
            topic_scores[topic] /= topic_counts[topic]

        top_topics = sorted(topic_scores, key=topic_scores.get, reverse=True)[:5]
        total_topic_score = sum(topic_scores[topic] for topic in top_topics)
        topic_probabilities = {topic: topic_scores[topic] / total_topic_score for topic in top_topics}

        filtered_indices = [i for i, topic in enumerate(sub_topics) if topic in top_topics]
        filtered_data = {
            'major_field': [sub_topics[i] for i in filtered_indices],
            'review': [reviews[i] for i in filtered_indices],
            'similarity_score': [similarities[i] for i in filtered_indices],
            'University': [universities[i] for i in filtered_indices],
            'professor': [professors[i] for i in filtered_indices],
            'course': [course[i] for i in filtered_indices],
            'created': [created_dates[i] for i in filtered_indices],
            'topic_probability': [topic_probabilities[sub_topics[i]] for i in filtered_indices]
        }

        filtered_df = pd.DataFrame(filtered_data)
        return filtered_df
    
    
    def evaluate_major(self, input_text, input_major):
        df = self.calculate_topic_probabilities_with_reviews(input_text)
        majors_prob_df = df.groupby('major_field')['topic_probability'].mean()
        sorted_majors = majors_prob_df.sort_values(ascending=False)

        ranking = None
        if input_major in sorted_majors.index:
            ranking = sorted_majors.index.tolist().index(input_major) + 1

        if ranking is None:
            assessment = 'bad'
        elif ranking <= 3:
            assessment = 'perfect'
        elif ranking <= 5:
            assessment = 'good'
        elif ranking <= 10:
            assessment = 'reasonable'
        else:
            assessment = 'bad'

        top_majors_list = sorted_majors.head(5).index.tolist()

        return assessment, top_majors_list
    def calculate_weighted_average_similarity(self,input_text):
        filtered_df = self.calculate_topic_probabilities_with_reviews(input_text)

        filtered_df['University'] = filtered_df['University'].astype(str)
        filtered_df['major_field'] = filtered_df['major_field'].astype(str)

        mean_scores = filtered_df.groupby(['University', 'major_field']).agg({'similarity_score': 'mean'}).reset_index()
        mean_scores['weighted_score'] = mean_scores.apply(
            lambda x: x['similarity_score'] * filtered_df[filtered_df['major_field'] == x['major_field']]['topic_probability'].iloc[0], 
            axis=1
        )
        was_scores = mean_scores.groupby('University')['weighted_score'].sum().reset_index().rename(columns={'weighted_score': 'WAS'})

        total_was = was_scores['WAS'].sum()
        was_scores['WAS'] = was_scores['WAS'] / total_was

        highest_prob_topic = mean_scores.loc[mean_scores.groupby('University')['similarity_score'].idxmax()][['University', 'major_field']]
        highest_prob_topic.columns = ['University', 'highest_prob_major']
        highest_prob_topic['University'] = highest_prob_topic['University'].astype(str)
        highest_prob_topic['highest_prob_major'] = highest_prob_topic['highest_prob_major'].astype(str)

        result = pd.merge(was_scores, highest_prob_topic, on='University')

        relevant_review = filtered_df.loc[filtered_df.groupby(['University', 'major_field'])['similarity_score'].idxmax()][['University', 'major_field', 'review','professor', 'course', 'created']]
        relevant_review.columns = ['University', 'highest_prob_major', 'Most_Relevant_Review', 'Most_Relevant_faculty', 'Most_Relevant_course', 'Most_Relevant_Created']
        relevant_review['University'] = relevant_review['University'].astype(str)
        relevant_review['highest_prob_major'] = relevant_review['highest_prob_major'].astype(str)

        final_result = pd.merge(result, relevant_review, on=['University', 'highest_prob_major'])

        return final_result[['University', 'WAS', 'highest_prob_major', 'Most_Relevant_Review','Most_Relevant_faculty', 'Most_Relevant_course', 'Most_Relevant_Created']]

    def calculate_confidence_score(self,input_text):
        df = self.calculate_topic_probabilities_with_reviews(input_text)    
        topic_counts = df['major_field'].value_counts()
        total_data_points = len(df)
        topic_portions = topic_counts / total_data_points
        topic_probabilities = df['topic_probability'].groupby(df['major_field']).mean()
        weighted_average = sum(topic_portions * topic_probabilities)
        return weighted_average

