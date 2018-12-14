# TensorFlow-lstm-char-cnn
Character-Aware Neural Language Models (Yoon Kim), lstm-char-cnn

## Paper
   * Character-Aware Neural Language Model: https://arxiv.org/abs/1508.06615
   <br/><br/>
   
## Dataset
   * PennTreebank(PTB)
   <br/><br/>
   
## Perplexity 
   * Paper large_test_ppl: 78.9
   * Paper small_test_ppl: 92.3  
   * My result <br/><br/>
      * large_test_ppl: 94.95
      * small_test_ppl: 97.59  <br/><br/>
   ![PPL](./result_image/perplexity.png)
   <br/><br/>
   
## Nearest neighbor words (based on cosine similarity) 
   * paper <br/><br/>
   ![p_cosine_similarity](./result_image/paper_cosine_similarity.PNG)<br/><br/>
   * large model <br/><br/>
   ![l_cosine_similarity](./result_image/Large_cosine_similarity.png)<br/><br/>
   * small model <br/><br/>
   ![s_cosine_similarity](./result_image/small_cosine_similarity.png)<br/><br/>
