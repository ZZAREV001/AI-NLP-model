using Tokenize
using DataLoaders

# Load source and target text data
src_texts = readlines("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/source.txt") 
trg_texts = readlines("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/target.txt")

# Create tokenizer 
tokenizer = Tokenizer(vocab_size=40000) 

# Fit tokenizer on combined data
all_texts = vcat(src_texts, trg_texts)
fit!(tokenizer, all_texts) 

# Convert source/target texts to sequences of indices
src_seqs = tokenize(tokenizer, src_texts)
trg_seqs = tokenize(tokenizer, trg_texts)

# Pad sequences to max length
src_seqs = padsequence(src_seqs, 512) 
trg_seqs = padsequence(trg_seqs, 512)

# Create data loader
data = DataLoader((src_seqs, trg_seqs), batch_size=64) 

# Preprocess functions
function preprocess(src_seq, trg_seq)

  # Convert to Tensor and add batch dimension
  src_seq = tensor(src_seq)[:, None, :] 
  trg_seq = tensor(trg_seq)[:, None, :]
  
  # Generate target input sequence 
  trg_input = trg_seq[:, :-1]
  
  # Generate target output sequence
  trg_target = trg_seq[:, 1:]  

  # Create mask for src padding
  src_pad_mask = src_seq .!= 0
  
  # Create mask for trg padding 
  trg_pad_mask = trg_input .!= 0
   
  return src_seq, src_pad_mask, trg_input, trg_pad_mask, trg_target

end

# Load and preprocess data
function process_data(src_path, trg_path, max_len; tokenizer=default_tokenizer)

  # Load text
  src_texts = readlines(src_path)
  trg_texts = readlines(trg_path)

  # Tokenize
  src_seqs = tokenize(tokenizer, src_texts)
  trg_seqs = tokenize(tokenizer, trg_texts)  

  # Pad sequences
  src_seqs = padsequence(src_seqs, max_len)
  trg_seqs = padsequence(trg_seqs, max_len)

  # Return DataLoader
  return DataLoader((src_seqs, trg_seqs), batch_size=64) 

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