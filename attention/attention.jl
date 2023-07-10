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
function attention_weights(q::Embedding, k::Embedding, v::Embedding; attn::Attention, mask=nothing)
    # Project queries, keys and values for each head
    qs = Array{Float32}(attn.n_heads, size(q, 2), attn.dim ÷ attn.n_heads) 
    ks = Array{Float32}(attn.n_heads, size(k, 2), attn.dim ÷ attn.n_heads)
    vs = Array{Float32}(attn.n_heads, size(v, 2), attn.dim ÷ attn.n_heads)
    
    for h = 1:attn.n_heads
        qs[h,:,:] = q * attn.wq[h]
        ks[h,:,:] = k * attn.wk[h]
        vs[h,:,:] = v * attn.wv[h]
    end
    
    # Compute attention weights   
    scores = qs * ks'                             
    scores .= scores ./ sqrt.(attn.dim ÷ attn.n_heads)
    
    if !isnothing(mask)
        scores .= scores .+ mask .* -1e9 
    end  
    probs = softmax(scores; dims=3)
    
    return probs
end  

# Apply attention weights  
function attention_output(probs::Array{Float32}, v::Embedding, attn::Attention)
  outputs = Array{Float32}(attn.n_heads, size(v,2),  attn.dim÷attn.n_heads)
  for h = 1:attn.n_heads
    outputs[h,:,:] = probs[h,:,:] * vs[h,:,:] 
  end
  output = reshape(reduce(hcat, outputs), :, attn.dim)
  
  return output
end  

end # module