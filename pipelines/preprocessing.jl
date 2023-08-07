module Preprocessing

using ..Data: pad_sequences, tokenize

function preprocess_data(data, tokenizer, max_sequence_length)
    # Tokenize the data
    sequences = tokenize(tokenizer, data)

    # Pad the sequences
    padded_sequences = pad_sequences(sequences, max_sequence_length)

    return padded_sequences
end

end # module
