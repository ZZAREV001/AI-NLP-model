An artificial intelligence program that is a natural language model. The AI program is based on the transformer model, reinforcement learning agents, and the new architecture called Receptance Weighted Key Value (RWKV - May 2023). It is specialized in time series. An important objective is to maintain linear time complexity for different kinds of datasets.

Parameters of our AI model (architecture RKWV - tranformer):

# Embeddings
src_vocab * dim = 40000 * 512 = 20,480,000
trg_vocab * dim = 40000 * 512 = 20,480,000

# Encoder 
n_layers * (
    4 * dim * dim * n_heads +   # Attention
    3 * dim * dim # Channel mixing
)  
= 6 * (4 * 512 * 512 * 8 + 3 * 512 * 512) 
= 49,152,000

# Decoder
n_layers * (
    4 * dim * dim * n_heads +   
    3 * dim * dim # Channel mixing
)
= 6 * (4 * 512 * 512 * 8 + 3 * 512 * 512)
= 49,152,000

Total parameters = 20,480,000 + 20,480,000 + 49,152,000 + 49,152,000
                 = 139,264,000 params