module Embeddings

"""
WordEmbeddings contains the embedding matrix for word embeddings.
"""
struct WordEmbeddings
    embeddings::Array{Float64,2}
    vocab_size::Int
    emb_size::Int
end

"""
PositionEmbeddings contains the embedding matrix for position embeddings.
"""
struct PositionEmbeddings
    embeddings::Array{Float64,2}
end 

"""
Lookup word embeddings for input words/tokens.
"""
function (we::WordEmbeddings)(x)
    return we.embeddings[x, :]
end

""" 
Add position embeddings to input.
"""
function (pe::PositionEmbeddings)(x)
    seq_len = size(x, 2)
    pos_embeddings = pe.embeddings[1:seq_len, :]
    return x + pos_embeddings 
end

end # module