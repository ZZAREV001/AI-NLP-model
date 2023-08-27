include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")

struct LinearAttention
    dim::Int 
    num_heads::Int
end
  
function LinearAttention(dim, num_heads)
    return LinearAttention(dim, num_heads) 
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

# Specific function to project the keys for multi-head attention
function project_heads(k::Embedding, attn::Attention)
    # Implementation of the projection (if needed)
end