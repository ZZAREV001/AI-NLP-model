module Embeddings

include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")

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

function WordEmbeddings(vocab_size, emb_size)
    embeddings = rand(Uniform(-1e-4, 1e-4), (vocab_size, emb_size)) 
    new(embeddings, vocab_size, emb_size)
end

end # module