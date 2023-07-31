module Attention

using ..Embeddings: Embedding, PositionEmbedding

"""
Attention layer. Computes attention weights and outputs a weighted sum of the values.
"""
struct Attention
    dim::Int         # Dimension of projections and embeddings
    n_heads::Int     # Number of attention heads      
end

# Compute attention weights
function linear_attention(k::Embedding, w; attn)

  # Project k for each head 
  ks = project_heads(k, attn) 
  
  # Apply time decay w
  ks = ks .* w

  # Compute attention scores
  scores = ks
  
  # Normalize 
  scores ./= sqrt.(attn.dim / attn.n_heads) 

  # Softmax
  probs = softmax(scores; dims=3)

  return probs 

end

end # module