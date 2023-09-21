module Validation

function validate_data(data, max_sequence_length::Int64)
    # Check that all sequences are the correct length
    for sequence in data
        if length(sequence) != max_sequence_length
            error("Sequence length does not match max_sequence_length")
        end
    end
end

end # module
