include("pipelines/ingestion.jl")
include("pipelines/preprocessing.jl")
include("pipelines/validation.jl")
include("models/transformer.jl")
include("RLAgentModule/rl_agent.jl") 
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")

using .Ingestion
using .Preprocessing
using .Validation
using .Models
using JSON

function main()

    # Load config
    config = JSON.parsefile("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/configs/config.json")  

    # Load data
    raw_data = Ingestion.load_data("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/vocab.json")

    # Debug: Print the type of raw_data
    println("Type of raw_data: ", typeof(raw_data))

    # Convert raw_data to a dictionary if it's not already one
    if typeof(raw_data) != Dict{String, Int64}
        # Assuming raw_data is a Vector{String}, create a dictionary mapping each word to an ID
        vocab_dict = Dict(word => i for (i, word) in enumerate(raw_data))
    else
        vocab_dict = raw_data
    end

    tokenizer = Tokenizer(vocab_dict, 40000, 0)  
    preprocessed_data = Preprocessing.preprocess_data(raw_data, tokenizer, 512)

    # Validate data
    Validation.validate_data(preprocessed_data, 512)

    # Initialize the required components for ReinforceRLAgent
    policy_layers = create_layers(Vector{Int}(config["policy_layers"]))
    value_layers = create_layers(Vector{Int}(config["value_layers"]))

    policy_net = Net(policy_layers)
    value_net = Net(value_layers)

    optimizer = SGDOptimizer(config["learning_rate"])
    
    n_layers = 6
    attn = LinearAttention(512, 8)
    ff = TransformerBlock(512, 2048)
    position = PositionalEncoding(512, 512)
    src_vocab = 40000
    trg_vocab = 40000
    encoder = TransformerEncoder(n_layers, attn, ff, position, src_vocab)
    decoder = TransformerDecoder(n_layers, attn, ff, position, trg_vocab)

    # Initialize ReinforceRLAgent
    rl_agent = Models.ReinforceRLAgent(net1, net2, optimizer, encoder, decoder)

    # Initialize model
    model = Models.Transformer(n_layers=6, n_heads=8, dim=512, dim_ff=2048, max_len=512, src_vocab=40000, trg_vocab=40000, rl_agent=rl_agent)

    # Train model
    for epoch in 1:100
        for (src, trg) in preprocessed_data
            output = Models.forward(model, src, trg, "teacher_forcing")
            # Compute loss and update model parameters
        end
    end

end

main()
