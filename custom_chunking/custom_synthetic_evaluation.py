from chunking_evaluation import SyntheticEvaluation
import sys
sys.path.append('..')  # Thêm đường dẫn cha vào sys.path để import utils
import json
import random
import os

import numpy as np
import pandas as pd
from utils import load_gemini_model, get_gemini_embedding
from chunking_evaluation.utils import rigorous_document_search
from chunking_evaluation.evaluation_framework.base_evaluation import BaseEvaluation
from importlib import resources


class GeminiSyntheticEvaluation(SyntheticEvaluation):
    def __init__(self, corpora_paths, queries_csv_path, chroma_db_path=None):
        # Chỉ khởi tạo những thứ cần thiết từ BaseEvaluation
        BaseEvaluation.__init__(self, questions_csv_path=queries_csv_path, chroma_db_path=chroma_db_path)  # optional, nếu BaseEvaluation ổn
        self.corpora_paths = corpora_paths
        self.questions_csv_path = queries_csv_path
        self.client = load_gemini_model()  # dùng Gemini client

        self.synth_questions_df = None

        # load prompt files
        # with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'prompts') as prompt_path:
        #     with open(os.path.join(prompt_path, 'question_maker_system.txt'), 'r') as f:
        #         self.question_maker_system_prompt = f.read()

        #     with open(os.path.join(prompt_path, 'question_maker_approx_system.txt'), 'r') as f:
        #         self.question_maker_approx_system_prompt = f.read()
            
        #     with open(os.path.join(prompt_path, 'question_maker_user.txt'), 'r') as f:
        #         self.question_maker_user_prompt = f.read()

        #     with open(os.path.join(prompt_path, 'question_maker_approx_user.txt'), 'r') as f:
        #         self.question_maker_approx_user_prompt = f.read()

        BASE_DIR = os.path.dirname(__file__)
        PROMPT_DIR = os.path.join(BASE_DIR, "prompts")  

        with open(os.path.join(PROMPT_DIR, "vie_question_maker_system.txt"), "r", encoding="utf-8") as f:
            self.question_maker_system_prompt = f.read()

        with open(os.path.join(PROMPT_DIR, "vie_question_maker_approx_system.txt"), "r", encoding="utf-8") as f:
            self.question_maker_approx_system_prompt = f.read()

        with open(os.path.join(PROMPT_DIR, "vie_question_maker_user.txt"), "r", encoding="utf-8") as f:
            self.question_maker_user_prompt = f.read()

        with open(os.path.join(PROMPT_DIR, "vie_question_maker_approx_user.txt"), "r", encoding="utf-8") as f:
            self.question_maker_approx_user_prompt = f.read()

    def _extract_question_and_approx_references(self, corpus, document_length=4000, prev_questions=[]):
        if len(corpus) > document_length:
            start_index = random.randint(0, len(corpus) - document_length)
            document = corpus[start_index : start_index + document_length]
        else:
            start_index = 0
            document = corpus
        
        if prev_questions is not None:
            if len(prev_questions) > 20:
                questions_sample = random.sample(prev_questions, 20)
                prev_questions_str = '\n'.join(questions_sample)
            else:
                prev_questions_str = '\n'.join(prev_questions)
        else:
            prev_questions_str = ""

        tagged_text, tag_indexes = self._tag_text(document)

        # completion = self.client.chat.completions.create(
        #     model="gpt-4-turbo",
        #     response_format={ "type": "json_object" },
        #     max_tokens=600,
        #     messages=[
        #         {"role": "system", "content": self.question_maker_approx_system_prompt},
        #         {"role": "user", "content": self.question_maker_approx_user_prompt.replace("{document}", tagged_text).replace("{prev_questions_str}", prev_questions_str)}
        #     ]
        # )
        # json_response = json.loads(completion.choices[0].message.content)

        # Using the generate_content method for Gemini 1.5
        completion = self.client.generate_content(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                self.question_maker_approx_system_prompt
                                + "\n\n"
                                + self.question_maker_approx_user_prompt
                                    .replace("{document}", tagged_text)
                                    .replace("{prev_questions_str}", prev_questions_str)
                            )
                        }
                    ]
                }
            ],
            generation_config={
                "max_output_tokens": 4000,
                "response_mime_type": "application/json"
            }
        )
        # Truy xuất JSON
        json_response = json.loads(completion.text)
        
        try:
            text_references = json_response['references']
        except KeyError:
            raise ValueError("The response does not contain a 'references' field.")
        try:
            question = json_response['question']
        except KeyError:
            raise ValueError("The response does not contain a 'question' field.")

        references = []
        for reference in text_references:
            reference_keys = list(reference.keys())

            if len(reference_keys) != 3:
                raise ValueError(f"Each reference must have exactly 3 keys: 'content', 'start_chunk', and 'end_chunk'. Got keys: {reference_keys}")

            if 'start_chunk' not in reference_keys or 'end_chunk' not in reference_keys:
                raise ValueError("Each reference must contain 'start_chunk' and 'end_chunk' keys.")

            if 'end_chunk' not in reference_keys:
                reference_keys.remove('content')
                reference_keys.remove('start_chunk')
                end_chunk_key = reference_keys[0]
                end_index = start_index + tag_indexes[reference[end_chunk_key]+1]
            else:
                end_index = start_index + tag_indexes[reference['end_chunk']+1]

            start_index = start_index + tag_indexes[reference['start_chunk']]
            references.append((corpus[start_index:end_index], start_index, end_index))
        
        return question, references
    
    def _extract_question_and_references(self, corpus, document_length=4000, prev_questions=[]):
        if len(corpus) > document_length:
            start_index = random.randint(0, len(corpus) - document_length)
            document = corpus[start_index : start_index + document_length]
        else:
            document = corpus
        
        if prev_questions is not None:
            if len(prev_questions) > 20:
                questions_sample = random.sample(prev_questions, 20)
                prev_questions_str = '\n'.join(questions_sample)
            else:
                prev_questions_str = '\n'.join(prev_questions)
        else:
            prev_questions_str = ""

        # completion = self.client.chat.completions.create(
        #     model="gpt-4-turbo",
        #     response_format={ "type": "json_object" },
        #     max_tokens=600,
        #     messages=[
        #         {"role": "system", "content": self.question_maker_system_prompt},
        #         {"role": "user", "content": self.question_maker_user_prompt.replace("{document}", document).replace("{prev_questions_str}", prev_questions_str)}
        #     ]
        # )
        
        # json_response = json.loads(completion.choices[0].message.content)

        # Using the generate_content method for Gemini 1.5
        completion = self.client.generate_content(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                self.question_maker_system_prompt
                                + "\n\n"
                                + self.question_maker_user_prompt
                                    .replace("{document}", document)
                                    .replace("{prev_questions_str}", prev_questions_str)
                            )
                        }
                    ]
                }
            ],
            generation_config={
                "max_output_tokens": 1200,
                "response_mime_type": "application/json"
            }
        )
        # Parse JSON từ completion.text
        json_response = json.loads(completion.text)
        
        try:
            text_references = json_response['references']
        except KeyError:
            raise ValueError("The response does not contain a 'references' field.")
        try:
            question = json_response['question']
        except KeyError:
            raise ValueError("The response does not contain a 'question' field.")

        references = []
        for reference in text_references:
            if not isinstance(reference, str):
                raise ValueError(f"Expected reference to be of type str, but got {type(reference).__name__}")
            target = rigorous_document_search(corpus, reference)
            if target is not None:
                reference, start_index, end_index = target
                references.append((reference, start_index, end_index))
            else:
                raise ValueError(f"No match found in the document for the given reference.\nReference: {reference}")
        
        return question, references
    
    def _get_sim(self, target, references):
        # response = self.client.embeddings.create(
        #     input=[target]+references,
        #     model="text-embedding-3-large"
        # )
        # nparray1 = np.array(response.data[0].embedding)
        
        valid_refs = [r for r in references if r.strip()]
        if not target or not target.strip() or not valid_refs:
            return [0.0] #fix
        
        response = get_gemini_embedding([target] + valid_refs)

        nparray1 = np.array(response[0])

        full_sim = []

        for i in range(1, len(response)): 
            nparray2 = np.array(response[i])
            cosine_similarity = np.dot(nparray1, nparray2) / (np.linalg.norm(nparray1) * np.linalg.norm(nparray2))
            full_sim.append(cosine_similarity)
            
        return full_sim

    def _corpus_filter_duplicates(self, corpus_id, synth_questions_df, threshold):
        corpus_questions_df = synth_questions_df[synth_questions_df['corpus_id'] == corpus_id].copy()

        count_before = len(corpus_questions_df)

        corpus_questions_df.drop_duplicates(subset='question', keep='first', inplace=True)

        questions = corpus_questions_df['question'].tolist()

        # response = self.client.embeddings.create(
        #     input=questions,
        #     model="text-embedding-3-large"
        # )

        # embeddings_matrix = np.array([data.embedding for data in response.data])
        embeddings_matrix = np.array(get_gemini_embedding(questions))
        dot_product_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)

        # Create a list of tuples containing the index pairs and their similarity
        similarity_pairs = [(i, j, dot_product_matrix[i][j]) for i in range(len(dot_product_matrix)) for j in range(i+1, len(dot_product_matrix))]

        # Sort the list of tuples based on the similarity in descending order
        similarity_pairs.sort(key=lambda x: x[2], reverse=True)

        similarity_scores = np.array([x[2] for x in similarity_pairs])

        most_similars = (dot_product_matrix - np.eye(dot_product_matrix.shape[0])).max(axis=1)

        def filter_vectors(sim_matrix, threshold):
            n = sim_matrix.shape[0]  # Number of vectors
            remaining = np.ones(n, dtype=bool)  # Initialize all vectors as remaining

            for i in range(n):
                if remaining[i] == 1:  # Only check for vectors that are still remaining
                    for j in range(i+1, n):
                        if remaining[j] == 1 and sim_matrix[i, j] > threshold:
                            remaining[j] = 0  # Remove vector j because it's too similar to vector i
            
            return remaining

        rows_to_keep = filter_vectors(dot_product_matrix, threshold)

        corpus_questions_df = corpus_questions_df[rows_to_keep]

        count_after = len(corpus_questions_df)

        print(f"Corpus: {corpus_id} - Removed {count_before - count_after} .")


        corpus_questions_df['references'] = corpus_questions_df['references'].apply(json.dumps)

        full_questions_df = pd.read_csv(self.questions_csv_path)
        full_questions_df = full_questions_df[full_questions_df['corpus_id'] != corpus_id]

        full_questions_df = pd.concat([full_questions_df, corpus_questions_df], ignore_index=True)
        # Drop the columns 'fixed', 'worst_ref_score' and 'diff_score' if they exist
        for col in ['fixed', 'worst_ref_score', 'diff_score']:
            if col in full_questions_df.columns:
                full_questions_df = full_questions_df.drop(columns=col)

        full_questions_df.to_csv(self.questions_csv_path, index=False)



    def _corpus_filter_poor_highlights(self, corpus_id, synth_questions_df, threshold):
        corpus_questions_df = synth_questions_df[synth_questions_df['corpus_id'] == corpus_id]

        def edit_row(row):
            question = row['question']
            references = [ref['content'] for ref in row['references']]

            # Debug: in ra ref nào rỗng
            empty_refs = [r for r in references if not r or not r.strip()]
            if empty_refs:
                print("⚠️ Empty reference found:", empty_refs, "for question:", question)
                
            similarity_scores = self._get_sim(question, references)
            worst_ref_score = min(similarity_scores)
            row['worst_ref_score'] = worst_ref_score
            return row

        # Apply the function to each row
        corpus_questions_df = corpus_questions_df.apply(edit_row, axis=1)

        count_before = len(corpus_questions_df)

        corpus_questions_df = corpus_questions_df[corpus_questions_df['worst_ref_score'] >= threshold]
        corpus_questions_df = corpus_questions_df.drop(columns=['worst_ref_score'])

        count_after = len(corpus_questions_df)

        print(f"Corpus: {corpus_id} - Removed {count_before - count_after} .")

        corpus_questions_df['references'] = corpus_questions_df['references'].apply(json.dumps)

        full_questions_df = pd.read_csv(self.questions_csv_path)
        full_questions_df = full_questions_df[full_questions_df['corpus_id'] != corpus_id]

        full_questions_df = pd.concat([full_questions_df, corpus_questions_df], ignore_index=True)
        # Drop the columns 'fixed', 'worst_ref_score' and 'diff_score' if they exist
        for col in ['fixed', 'worst_ref_score', 'diff_score']:
            if col in full_questions_df.columns:
                full_questions_df = full_questions_df.drop(columns=col)

        full_questions_df.to_csv(self.questions_csv_path, index=False)


    def filter_poor_excerpts(self, threshold=0.36, corpora_subset=[]):
        if os.path.exists(self.questions_csv_path):
            synth_questions_df = pd.read_csv(self.questions_csv_path)
            if len(synth_questions_df) > 0:
                synth_questions_df['references'] = synth_questions_df['references'].apply(json.loads)
                corpus_list = synth_questions_df['corpus_id'].unique().tolist()
                if corpora_subset:
                    corpus_list = [c for c in corpus_list if c in corpora_subset]
                for corpus_id in corpus_list:
                    self._corpus_filter_poor_highlights(corpus_id, synth_questions_df, threshold)
