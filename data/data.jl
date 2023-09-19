module Data

using TextAnalysis

# Define the tokenizer
function my_tokenizer(text)
  return ngrams(text, 1) # 1-grams, i.e., individual words
end

default_tokenizer = my_tokenizer

# Load source and target text data
src_texts = readlines("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/source.txt") 
trg_texts = readlines("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/target.txt")

# Create a StringDocument from the texts
all_texts = vcat(src_texts, trg_texts)
all_docs = StringDocument.(all_texts)

# Create a Corpus from the documents
corpus = Corpus(all_docs)

# Tokenize the corpus
src_seqs = [ngrams(text, 1) for text in src_texts]
trg_seqs = [ngrams(text, 1) for text in trg_texts]

# Convert source/target texts to sequences of indices
# Note: possible to write custom code to convert tokens to indices based on your vocabulary
src_seqs = [convert_to_indices(tokenize(tokenizer, text)) for text in src_texts]
trg_seqs = [convert_to_indices(tokenize(tokenizer, text)) for text in trg_texts]

function padsequence(sequences, max_len; padding_value=0)
  padded_sequences = []
  for seq in sequences
      pad_length = max_len - length(seq)
      padded_seq = vcat(seq, fill(padding_value, pad_length))
      push!(padded_sequences, padded_seq)
  end
  return padded_sequences
end

# Preprocess functions
function preprocess(src_seq, trg_seq)

  # Convert to Tensor and add batch dimension
  src_seq = unsqueeze(tensor(src_seq), 2) # Add a new dimension at the second position
  trg_seq = unsqueeze(tensor(trg_seq), 2)
  
  # Generate target input sequence 
  trg_input = trg_seq[:, 1:end-1, :]
  
  # Generate target output sequence
  trg_target = trg_seq[:, 2:end, :]  

  # Create mask for src padding
  src_pad_mask = src_seq .!= 0
  
  # Create mask for trg padding 
  trg_pad_mask = trg_input .!= 0
   
  return src_seq, src_pad_mask, trg_input, trg_pad_mask, trg_target

end

# Load and preprocess data
function process_data(src_path, trg_path, max_len; tokenizer, batchsize=64)

  # Load text
  src_texts = readlines(src_path)
  trg_texts = readlines(trg_path)

  # Tokenize
  src_seqs = [tokenizer(text) for text in src_texts]
  trg_seqs = [tokenizer(text) for text in trg_texts]

  # Pad sequences
  src_seqs = padsequence(src_seqs, max_len)
  trg_seqs = padsequence(trg_seqs, max_len)

  # Combine source and target sequences
  data = [(src, trg) for (src, trg) in zip(src_seqs, trg_seqs)]

  # Create batches
  batches = [data[i:min(i+batchsize-1, end)] for i in 1:batchsize:length(data)]

  return batches
end

# Postprocess functions 
function postprocess(pred, target)

  # Flatten prediction and target
  pred = flatten(pred, dims=(1,2)) 
  target = flatten(target, dims=(1,2))

  # Ignore padding tokens 
  mask = target .!= 0

  # Compute cross entropy loss
  loss = crossentropy(pred[mask], target[mask])

  # Return average loss
  return mean(loss)

end

end
