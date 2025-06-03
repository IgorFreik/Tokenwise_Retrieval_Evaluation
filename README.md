# Benchmarking Chunking Strategies

## Study Goal

This study aims to evaluate and compare different chunking strategies for text retrieval systems, focusing on the impact of chunk size and number of retrieved chunks on retrieval performance. The primary objective is to identify optimal configurations that balance retrieval accuracy (measured through Recall, Precision, and IoU) with computational efficiency.


## Methodology

The study employs a systematic approach to evaluate different chunking strategies and their impact on retrieval performance. The methodology encompasses several key components:

1. **Chunking Techniques Analysis**:
   - Basic/recursive character chunking
   - HTML/markdown/code-specific chunking
   - Semantic chunking approaches
   - Semantic double merging chunking
   - Advanced strategies including:
     * Semantic clustering chunking
     * LLM-based semantic chunking
     * Kamradt chunking

2. **Evaluation Framework**:
   - Dataset Construction:
     * 
   
   - Performance Metrics:
     * Token-wise Recall: Measures the proportion of relevant tokens retrieved
     * Token-wise Precision: Measures the accuracy of retrieved tokens
     * Intersection over Union (IoU): Provides a balanced measure of retrieval performance
     * Execution time: Monitored to assess computational efficiency
   
   - Visualization and Analysis:
     * Trade-off analysis between chunk size and retrieval quality

3. **Parameter Space Exploration**:
   - Varied chunk sizes (ranging from 25 to 300 tokens)
   - Tested different numbers of retrieved chunks (from 1 to 10)
   - Used zero chunk overlap for consistency
   - Evaluated impact of different content types (code, text, tables)

4. **Quality Assessment**:
   - Semantic completeness evaluation
   - Context preservation analysis
   - Information loss prevention
   - Retrieval accuracy across different content types

## Results

The evaluation results are presented using token-wise Recall, Precision, and Intersection over Union (IoU), as defined in the linked paper by Brandon Smith and Anton Troynikov. All results are based on a chunking overlap of zero, though this parameter may be explored in future work.

The analysis indicates that the best-performing configuration in terms of IoU occurs when the chunk size is set to 50 and the number of retrieved chunks is 1. This setting also yields the highest precision, with recall and precision values of 0.375 and 0.339, respectively. hese observations are drawn from the IoU heatmap in the results folder.

As expected, recall consistently increases with the number of retrieved chunks. A similar pattern is observed for chunk size, suggesting that adding more text per chunk does not negatively impact the embedding model's ability to identify relevant chunks. The recall values range from 0.164 to 0.982. The highest recall with the lowest total number of retrieved tokens (calculated as the product of chunk size and the number of retrieved chunks) occurs at num_retrieved_chunks = 7 and chunk_size = 300 tokens.

Precision generally follows a monotonous trend, with one exception: chunk_size = 25 and num_retrieved_chunks = 1 produce lower precision than chunk_size = 50 and num_retrieved_chunks = 1. This suggests that, in most cases, increasing chunk size has a more significant positive effect than any potential recall improvement. However, for chunk_size = 1 and num_retrieved_chunks = 1, the opposite effect is likely at play, as the average number of tokens in the reference answers (68.4 tokens) is significantly larger than the retrieved token count.

Regarding retrieval efficiency, retrieval time increases with the number of retrieved chunks, peaking at approximately +20% when comparing 10 retrieved chunks to 1 retrieved chunk. However, chunk size does not exhibit a clear correlation with retrieval time. These observations are drawn from execution time plots.

The dataset with computed metrics is available in the results folder:
results/results_250331_140551/hyperparameter_results_token_wise.csv
