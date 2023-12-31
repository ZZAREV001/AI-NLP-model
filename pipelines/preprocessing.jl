module Preprocessing

include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/data.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/RLAgentModule/rl_agent.jl")

# Define the pad_sequences function
function pad_sequences(sequences, max_sequence_length)
    # Initialize an array to hold the padded sequences
    padded_sequences = []

    # Loop through each sequence in the input
    for seq in sequences
        # If the sequence is shorter than max_sequence_length, pad it with zeros
        if length(seq) < max_sequence_length
            pad_length = max_sequence_length - length(seq)
            padded_seq = vcat(seq, zeros(pad_length))
        # If the sequence is longer than max_sequence_length, truncate it
        elseif length(seq) > max_sequence_length
            padded_seq = seq[1:max_sequence_length]
        # If the sequence is exactly max_sequence_length, use it as is
        else
            padded_seq = seq
        end

        # Add the padded/truncated sequence to the array of padded sequences
        push!(padded_sequences, padded_seq)
    end

    return padded_sequences
end

# Define the tokenize function
function tokenize(tokenizer, data)
    # Initialize an array to hold the tokenized sequences
    tokenized_sequences = []

    # Loop through each piece of data in the input
    for text in data
        # Split the text into words using a simple whitespace-based tokenization
        tokens = split(text)
        
        # Convert each token to its corresponding ID in the tokenizer's vocabulary
        token_ids = [get(tokenizer.vocab, token, tokenizer.unk_token_id) for token in tokens]
        println("Debug in tokenize: Type of tokenizer: ", typeof(tokenizer))
        println("Debug in tokenize: Content of tokenizer: ", tokenizer)
        # Add the sequence of token IDs to the array of tokenized sequences
        push!(tokenized_sequences, token_ids)
    end

    return tokenized_sequences
end

function preprocess_data(data, tokenizer, max_sequence_length::Int64)
    
    # Tokenize the data
    sequences = tokenize(tokenizer, data)
    println("Debug in preprocess_data: Type of tokenizer: ", typeof(tokenizer))
    println("Debug in preprocess_data: Content of tokenizer: ", tokenizer)

    # Pad the sequences
    padded_sequences = pad_sequences(sequences, max_sequence_length)

    return padded_sequences
end

# Define the process_data function
function process_data(src_path::String, trg_path::String, max_sequence_length::Int64, tokenizer)
    # Read data from source and target files
    src_data = readlines(src_path)
    trg_data = readlines(trg_path)

    # Tokenize and pad the source data
    src_sequences = preprocess_data(src_data, tokenizer, max_sequence_length)

    # Tokenize and pad the target data
    trg_sequences = preprocess_data(trg_data, tokenizer, max_sequence_length)

    # Combine the processed source and target data into a single data structure
    # This could be a tuple, dictionary, or any other structure (we change here)
    processed_data = (src_sequences, trg_sequences)

    return processed_data
end


end # module
