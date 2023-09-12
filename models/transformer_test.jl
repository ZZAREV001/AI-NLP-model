include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")
using NNlib: relu

# Define the vocabulary
struct Vocabulary
    words::Vector{String}
end

# Instanciation of the RLAgent
rl_agent = ReinforceRLAgent(
    Net([Layer(rand(2, 2), rand(2), relu)]), 
    Net([Layer(rand(2, 2), rand(2), relu)]), 
    SGDOptimizer(0.01), 
    TransformerEncoder(6, Attention(512, 8), FeedForward(rand(512, 512), rand(512, 512)), PositionEncoding(rand(512, 100))), 
    TransformerDecoder(6, Attention(512, 8), FeedForward(rand(512, 512), rand(512, 512)), PositionEncoding(rand(512, 100)), Attention(512, 8))
)

# Define input parameters for the Transformer function
n_layers = 6
n_heads = 8
dim = 512
dim_ff = 2048
max_len = 100
src_vocab = Vocabulary(["hello", "world"])
trg_vocab = Vocabulary(["bonjour", "monde"])

# Create instances of the necessary components
attn = Attention(dim, n_heads)
ff = FeedForward(rand(dim, dim_ff), rand(dim_ff, dim))
position_encoding = PositionEncoding(rand(dim, max_len))

encoder = TransformerEncoder(n_layers, attn, ff, position_encoding) # Removed src_vocab
decoder = TransformerDecoder(n_layers, attn, ff, position_encoding, Attention(dim, n_heads)) # Removed trg_vocab

# Create the Transformer instance using the components
transformer = Transformer(encoder, decoder, rl_agent)

# Check the output of the Transformer function
function Base.show(io::IO, t::Transformer)
    println(io, "Transformer object:")
    println(io, "  - Encoder: ", summary(t.encoder))
    println(io, "  - Decoder: ", summary(t.decoder))
    println(io, "  - RL Agent: ", summary(t.rl_agent))
end

# Log the output of the encoder and decoder
LoggingModule.log_encoder_decoder_output(transformer.encoder_output, transformer.decoder_output)

# Log the output of the RL agent
LoggingModule.log_rl_agent_output(transformer.rl_agent_output)

# Log the output of the Transformer function
LoggingModule.log_transformer_output(transformer.output)

# Log the input to the Transformer function
LoggingModule.log_transformer_input(transformer.input)