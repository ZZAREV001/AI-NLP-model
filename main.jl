include("pipelines/ingestion.jl")
include("pipelines/preprocessing.jl")
include("pipelines/validation.jl")
include("models/transformer.jl")
include("models/rl_agent.jl")

using .Ingestion
using .Preprocessing
using .Validation
using .Models

function main()

    # Load data
    raw_data = Ingestion.load_data("data.txt")

    # Preprocess data
    tokenizer = Tokenizer(vocab_size=40000) 
    preprocessed_data = Preprocessing.preprocess_data(raw_data, tokenizer, max_sequence_length=512)

    # Validate data
    Validation.validate_data(preprocessed_data, max_sequence_length=512)

    # Initialize model
    model = Models.Transformer(n_layers=6, n_heads=8, dim=512, dim_ff=2048, max_len=512, src_vocab=40000, trg_vocab=40000, rl_agent=Models.ReinforceRLAgent())

    # Train model
    for epoch in 1:100
        for (src, trg) in preprocessed_data
            output = Models.forward(model, src, trg, "teacher_forcing")
            # Compute loss and update model parameters
        end
    end

end

main()
