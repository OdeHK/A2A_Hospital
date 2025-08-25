from chunking_evaluation.evaluation_framework.base_evaluation import BaseEvaluation
import chromadb
from importlib import resources
import os
import numpy as np

class CustomBaseEvaluation(BaseEvaluation):
    def _chunker_to_collection(self, chunker, embedding_function, chroma_db_path:str = None, collection_name:str = None):
        collection = None

        if chroma_db_path is not None:
            try:
                chunk_client = chromadb.PersistentClient(path=chroma_db_path)
                collection = chunk_client.create_collection(collection_name, embedding_function=embedding_function, metadata={"hnsw:search_ef":50})
                print("Created collection: ", collection_name)
            except Exception as e:
                print("Failed to create collection: ", e)
                pass
                # This shouldn't throw but for whatever reason, if it does we will default to below.

        collection_name = "auto_chunk"
        if collection is None:
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception as e:
                pass
            collection = self.chroma_client.create_collection(collection_name, embedding_function=embedding_function, metadata={"hnsw:search_ef":50})

        docs, metas = self._get_chunks_and_metadata(chunker)

        BATCH_SIZE = 500
        for i in range(0, len(docs), BATCH_SIZE):
            batch_docs = docs[i:i+BATCH_SIZE]
            batch_metas = metas[i:i+BATCH_SIZE]
            batch_ids = [str(i) for i in range(i, i+len(batch_docs))]
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )

            # print("Documents: ", batch_docs)
            # print("Metadatas: ", batch_metas)

        return collection
    
    def run(self, chunker, embedding_function=None, retrieve:int = 5, db_to_save_chunks: str = None):
        """
        This function runs the evaluation over the provided chunker.

        Parameters:
        chunker: The chunker to evaluate.
        embedding_function: The embedding function to use for calculating the nearest neighbours during the retrieval step. If not provided, the default OpenAI embedding function is used.
        retrieve: The number of chunks to retrieve per question. If set to -1, the function will retrieve the minimum number of chunks that contain excerpts for a given query. This is typically around 1 to 3 but can vary by question. By setting a specific value for retrieve, this number is fixed for all queries.
        """
        self._load_questions_df()
        if embedding_function is None:
            embedding_function = get_openai_embedding_function()

        collection = None
        if db_to_save_chunks is not None:
            chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else "0"
            chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else "0"
            embedding_function_name = embedding_function.__class__.__name__
            if embedding_function_name == "SentenceTransformerEmbeddingFunction":
                embedding_function_name = "SentEmbFunc"
            collection_name = embedding_function_name + '_' + chunker.__class__.__name__ + '_' + str(int(chunk_size)) + '_' + str(int(chunk_overlap))
            try:
                chunk_client = chromadb.PersistentClient(path=db_to_save_chunks)
                collection = chunk_client.get_collection(collection_name, embedding_function=embedding_function)
            except Exception as e:
                # Get collection throws if the collection does not exist. We will create it below if it does not exist.
                collection = self._chunker_to_collection(chunker, embedding_function, chroma_db_path=db_to_save_chunks, collection_name=collection_name)

        if collection is None:
            collection = self._chunker_to_collection(chunker, embedding_function)

        question_collection = None

        if self.is_general:
            with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'general_evaluation_data') as general_benchmark_path:
                questions_client = chromadb.PersistentClient(path=os.path.join(general_benchmark_path, 'questions_db'))
                if embedding_function.__class__.__name__ == "OpenAIEmbeddingFunction":
                    try:
                        if embedding_function._model_name == "text-embedding-3-large":
                            question_collection = questions_client.get_collection("auto_questions_openai_large", embedding_function=embedding_function)
                        elif embedding_function._model_name == "text-embedding-3-small":
                            question_collection = questions_client.get_collection("auto_questions_openai_small", embedding_function=embedding_function)
                    except Exception as e:
                        print("Warning: Failed to use the frozen embeddings originally used in the paper. As a result, this package will now generate a new set of embeddings. The change should be minimal and only come from the noise floor of OpenAI's embedding function. The error: ", e)
                elif embedding_function.__class__.__name__ == "SentenceTransformerEmbeddingFunction":
                    try:
                        question_collection = questions_client.get_collection("auto_questions_sentence_transformer", embedding_function=embedding_function)
                    except:
                        print("Warning: Failed to use the frozen embeddings originally used in the paper. As a result, this package will now generate a new set of embeddings. The change should be minimal and only come from the noise floor of SentenceTransformer's embedding function. The error: ", e)
        
        if not self.is_general or question_collection is None:
            # if self.is_general:
            #     print("FAILED TO LOAD GENERAL EVALUATION")
            try:
                self.chroma_client.delete_collection("auto_questions")
            except Exception as e:
                pass
            question_collection = self.chroma_client.create_collection("auto_questions", embedding_function=embedding_function, metadata={"hnsw:search_ef":50})
            question_collection.add(
                documents=self.questions_df['question'].tolist(),
                metadatas=[{"corpus_id": x} for x in self.questions_df['corpus_id'].tolist()],
                ids=[str(i) for i in self.questions_df.index]
            )
        
        question_db = question_collection.get(include=['embeddings'])

        # Convert ids to integers for sorting
        question_db['ids'] = [int(id) for id in question_db['ids']]
        # Sort both ids and embeddings based on ids
        _, sorted_embeddings = zip(*sorted(zip(question_db['ids'], question_db['embeddings'])))

        # Sort questions_df in ascending order
        self.questions_df = self.questions_df.sort_index()

        brute_iou_scores, highlighted_chunks_count = self._full_precision_score(collection.get()['metadatas'])

        if retrieve == -1:
            maximum_n = min(20, max(highlighted_chunks_count))
        else:
            highlighted_chunks_count = [retrieve] * len(highlighted_chunks_count)
            maximum_n = retrieve

        # arr_bytes = np.array(list(sorted_embeddings)).tobytes()
        # print("Hash: ", hashlib.md5(arr_bytes).hexdigest())

        # Retrieve the documents based on sorted embeddings
        retrievals = collection.query(query_embeddings=list(sorted_embeddings), n_results=maximum_n)

        iou_scores, recall_scores, precision_scores = self._scores_from_dataset_and_retrievals(retrievals['metadatas'], highlighted_chunks_count)


        corpora_scores = {

        }
        for index, row in self.questions_df.iterrows():
            if row['corpus_id'] not in corpora_scores:
                corpora_scores[row['corpus_id']] = {
                    "precision_omega_scores": [],
                    "iou_scores": [],
                    "recall_scores": [],
                    "precision_scores": []
                }
            
            corpora_scores[row['corpus_id']]['precision_omega_scores'].append(brute_iou_scores[index])
            corpora_scores[row['corpus_id']]['iou_scores'].append(iou_scores[index])
            corpora_scores[row['corpus_id']]['recall_scores'].append(recall_scores[index])
            corpora_scores[row['corpus_id']]['precision_scores'].append(precision_scores[index])


        brute_iou_mean = np.mean(brute_iou_scores)
        brute_iou_std = np.std(brute_iou_scores)

        recall_mean = np.mean(recall_scores)
        recall_std = np.std(recall_scores)

        iou_mean = np.mean(iou_scores)
        iou_std = np.std(iou_scores)

        precision_mean = np.mean(precision_scores)
        precision_std = np.std(precision_scores)

        # print("Recall scores: ", recall_scores)
        # print("Precision scores: ", precision_scores)
        # print("Recall Mean: ", recall_mean)
        # print("Precision Mean: ", precision_mean)

        return {
            "corpora_scores": corpora_scores,
            "iou_mean": iou_mean,
            "iou_std": iou_std,
            "recall_mean": recall_mean,
            "recall_std": recall_std,
            "precision_omega_mean": brute_iou_mean,
            "precision_omega_std": brute_iou_std,
            "precision_mean": precision_mean,
            "precision_std": precision_std
        }