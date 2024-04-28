include("pipelines/ingestion.jl")
include("pipelines/preprocessing.jl")
include("pipelines/validation.jl")
include("models/transformer.jl")
include("RLAgentModule/rl_agent.jl") 
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/training/train.jl")

using .Ingestion
using .Preprocessing
using .Validation
using .Models
using JSON

function main()

    # Load config
    config = JSON.parsefile("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/configs/config.json")  

    # Load data
    raw_data = Ingestion.load_data("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/vocab2.json")

    # Debug: Print the type of raw_data
    println("Type of raw_data: ", typeof(raw_data))
    println("Length of raw_data: ", length(raw_data))
    println("Length of preprocessed_data: ", length(preprocessed_data))

    # Convert raw_data to a dictionary if it's not already one
    if typeof(raw_data) != Dict{String, Int}
        # Assuming raw_data is a Vector{String}, create a dictionary mapping each word to an ID
        vocab_dict = Dict(word => i for (i, word) in enumerate(raw_data))
    else
        vocab_dict = raw_data
    end

    tokenizer = Tokenizer(vocab_dict, 40000, 0)  
    # Debug lines to print the type and fields of 'tokenizer'
    println("Type of tokenizer: ", typeof(tokenizer))
    println("Fields of tokenizer: ", fieldnames(typeof(tokenizer)))
    
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

    # A function to tokenize the input question
    function tokenize_input(question, tokenizer)
        # Tokenize the input question
        tokens = tokenizer.tokenize(question)
        
        # Convert tokens to indices based on the vocabulary
        indices = [tokenizer.vocab_dict[token] for token in tokens]
        
        return indices
    end

    # Train model
    for epoch in 1:50
        for (src, trg) in preprocessed_data
            output = Models.forward(model, src, trg, "teacher_forcing")
            if isempty(src) || isempty(trg)
                println("Empty batch detected.")
                continue
            end            
            # Compute loss and update model parameters
        end
    end

    # Inference step
    while true
        println("Enter a question (or type 'quit' to exit):")
        question = readline()
        
        if question == "quit"
            break
        end
        
        # Tokenize the input question
        input_indices = tokenize_input(question, tokenizer)
        
        # Perform inference
        output_indices = Models.predict(model, input_indices)
        
        # Convert output indices back to words
        output_words = [tokenizer.id_to_word(idx) for idx in output_indices]
        
        # Print the generated answer
        println("Generated answer: ", join(output_words, " "))
    end

end

main()
