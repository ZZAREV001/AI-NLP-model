# Common struct definitions and functions used by several modules
"""
WordEmbeddings contains the embedding matrix for word embeddings.
"""
struct WordEmbeddings
    embeddings::Array{Float64,2}
    vocab_size::Int
    emb_size::Int
end

"""
PositionEmbeddings contains the embedding matrix for position embeddings.
"""
struct PositionEmbeddings
    embeddings::Array{Float64,2}
end 

struct Embedding
    weights::Array{Float64,2} # Embedding weights
    vocab_size::Int           # Vocabulary size
    emb_size::Int             # Embedding dimension
end



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
