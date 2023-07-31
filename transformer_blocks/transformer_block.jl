module TransformerBlocks

using ..Attention, ..Embeddings

"""
TransformerBlock contains core RWKV modifications - linear attention in Attention module, channel mixing, and token shift before attention.
"""
struct TransformerBlock
    attention::Attention
end

function (tb::TransformerBlock)(x)
    # Token shift
    x_shifted = cat(x[:, 2:end, :], x[:, 1:1, :], dims=2)

    # Linear mixing 
    μ = 0.5
    x_mixed = μ .* x + (1 - μ) .* x_shifted
    
    x = tb.attention(x_shifted)
    
    # Channel mixing
    r = sigmoid(W_r * x)
    k = relu(W_k * x)
    v = W_v * max.(k, 0).^2
    x = r .* v

    return x 
end

# Small embeddings
embed_init = Uniform(-1e-4, 1e-4)

# Identity-style initialization
linear_init = GlorotUniform()

end # module