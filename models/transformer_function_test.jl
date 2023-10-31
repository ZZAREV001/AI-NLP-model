using Test
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/models/transformer.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/attention/attention.jl")

# Initialize Attention
attention_instance = attention.Attention(attention.Linear(32, 32), attention.Linear(32, 32), attention.Linear(32, 32))

# Initialize FeedForward
feed_forward_instance = FeedForward(attention.Linear(32, 64), attention.Linear(64, 32))

# Initialize PositionEncoding
position_encoding_instance = PositionEncoding(100, 32)

# Initialize TransformerEncoder and TransformerDecoder
encoder_instance = TransformerEncoder(2, attention_instance, feed_forward_instance, position_encoding_instance)
decoder_instance = TransformerDecoder(2, attention_instance, feed_forward_instance, position_encoding_instance)

# Initialize RLAgent
policy_net_instance = Net([Layer(32, 64), Layer(64, 32)])
value_net_instance = Net([Layer(32, 64), Layer(64, 32)])
optimizer_instance = SGDOptimizer(0.001)
rl_agent_instance = ReinforceRLAgent(policy_net_instance, value_net_instance, optimizer_instance)

@testset "Transformer Tests" begin
    @test typeof(Transformer(encoder_instance, decoder_instance, rl_agent_instance)) == typeof(YourTransformerType)
    # Add more tests
end