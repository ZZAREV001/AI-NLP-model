# Testing of the forward function
using Random

# Initialize parameters
vocab_size = 40000
emb_size = 512
max_len = 100
n_layers = 6
n_heads = 8
dim = 512
dim_ff = 2048

# Create instances of the necessary components
word_embeddings = WordEmbeddings(rand(Float64, vocab_size, emb_size), vocab_size, emb_size)
position_embeddings = PositionEmbeddings(rand(Float64, max_len, emb_size))
embedding = Embedding(rand(Float64, vocab_size, emb_size), vocab_size, emb_size)
attention = Attention(dim, n_heads)
feed_forward = FeedForward(rand(Float64, dim, dim_ff), rand(Float64, dim_ff, dim))
position_encoding = PositionEncoding(rand(Float64, dim, max_len))
encoder = TransformerEncoder(n_layers, attention, feed_forward, position_encoding)
decoder = TransformerDecoder(n_layers, attention, feed_forward, position_encoding, attention)
layer = Layer(rand(Float64, dim, dim), rand(Float64, dim), relu)
net = Net([layer, layer])
optimizer = SGDOptimizer(0.01)
rl_agent = ReinforceRLAgent(net, net, optimizer, encoder, decoder)
transformer = Transformer(encoder, decoder, rl_agent)

# Logging functions
function log_encoder_decoder_output(encoder_output, decoder_output)
    println("Logging encoder and decoder output...")
end

function log_attention_weights(attn_weights)
    println("Logging attention weights...")
end

# Call the forward function
output = forward(transformer, src, trg, "teacher_forcing")
println("Output: ", output)

