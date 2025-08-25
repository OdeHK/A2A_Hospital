
from custom_chunking import GeminiSyntheticEvaluation


corpora_paths = [
    'cleaned_data/page_301_400.txt',
    'cleaned_data/page_401_500.txt',
    'cleaned_data/page_501_600.txt',
    'cleaned_data/page_601_700.txt',
    'cleaned_data/page_701_800.txt',
    'cleaned_data/page_801_900.txt',
    'cleaned_data/page_901_1000.txt',
    'cleaned_data/page_1001_1100.txt',
    'cleaned_data/page_1101_1123.txt',
]
queries_csv_path = 'generated_queries_and_excerpts_2.csv'


eval = GeminiSyntheticEvaluation(
    corpora_paths=corpora_paths,
    queries_csv_path=queries_csv_path,
)
eval.generate_queries_and_excerpts(approximate_excerpts=True, num_rounds=1, queries_per_corpus=50)