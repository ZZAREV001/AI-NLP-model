module TransformerBlocks

using ..Attention, ..Embeddings

"""
TransformerBlock contains one attention layer and one feedforward layer.
"""
struct TransformerBlock
    attention::Attention.Attention
    feedforward::Feedforward    
end

"""
Feedforward layer applies a linear transformation (defined by dimension and activation) 
and dropout.
"""
struct Feedforward
    input_dim::Int
    output_dim::Int
    activation::Function
    dropout::Float64
end

function (ff::Feedforward)(x)
    x = Linear(ff.input_dim, ff.output_dim)(x)
    x = ff.activation.(x)
    x = Dropout(ff.dropout)(x)
    return x
end

function (tb::TransformerBlock)(x)
    x = tb.attention(x)
    x = tb.feedforward(x)
    return x
end

end # module