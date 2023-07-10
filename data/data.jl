module Data

"""
Vocabulary mapping from words to indices.
"""
struct Vocabulary
    word2idx::Dict{String,Int}
    idx2word::Dict{Int,String}
end 

"""
Pad/truncate sequences to max_sequence_length.
""" 
function pad_sequences(sequences, max_sequence_length)
    # Pad/truncate sequences 
end

"""
Return a DataLoader over the training set.
"""
function get_train_loader(batch_size)
    # Load training data and tokenize 
    sequences = # Load and tokenize text into word sequences 
    # Pad sequences
    padded = pad_sequences(sequences, max_sequence_length)    
    # Create DataLoader
    return DataLoader(padded, batch_size=batch_size, ...) 
end

# Similarly for val_loader and test_loader

end # module