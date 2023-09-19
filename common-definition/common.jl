# Common struct definitions and functions used by several modules
"""
Tokenizer contains the vocabulary and methods to convert text data into a format that's usable by the model.
"""
struct Tokenizer
    vocab::Dict{String, Int}  # Dictionary mapping tokens to their respective IDs
    unk_token_id::Int         # ID to use for unknown tokens
    vocab_size::Int           # The size of the vocabulary
end

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

struct FeedForward
    layer1::Array{Float64,2}
    layer2::Array{Float64,2}
end

struct PositionEncoding
    encoding::Array{Float64,2}
end

struct TransformerEncoder
    n_layers::Int
    attention::Attention
    feed_forward::FeedForward
    position_encoding::PositionEncoding
end

struct TransformerDecoder
    n_layers::Int
    attention::Attention
    feed_forward::FeedForward
    position_encoding::PositionEncoding
    encoder_attention::Attention
end

"""
Abstract RL agent type
"""
abstract type RLAgent end

struct Layer
    weights::Array{Float64,2}
    biases::Vector{Float64}
    activation::Function
end

struct Net
    layers::Vector{Layer}
end

struct SGDOptimizer
    learning_rate::Float64
end

struct ReinforceRLAgent <: RLAgent
    policy_net::Net   
    value_net::Net   
    optimizer::SGDOptimizer
    encoder::TransformerEncoder
    decoder::TransformerDecoder
end

struct Transformer
    encoder::TransformerEncoder
    decoder::TransformerDecoder
    rl_agent::RLAgent 
end
