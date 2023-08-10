include("data.jl")

train_data = process_data("train_src.txt", "train_trg.txt", 512)
val_data = process_data("val_src.txt", "val_trg.txt", 512) 

# Extract text from PDFs
text = ""
for pdf in pdf_files
    text += PDFPlumber.extract_text(pdf) 
end

# Preprocess text
text = lowercase(text)
text = remove_punctuation(text) 

# Split into sentences 
sentences = split_sentences(text)

# Create train/val/test splits
train, val, test = split_data(sentences) 

# Vectorize sentences into input/target tensors 
input_tensor, target_tensor = vectorize_data(train)

# Define model, optimizer, loss 
model = TransformerModel()
optimizer = Adam(0.01) 
loss_fn = CrossEntropy()

# Generate model input and targets 
train_input, train_target = vectorize_data(train_sentences)

# Wrap input and target in DataLoader
train_data = DataLoader((train_input, train_target), batchsize=64) 

# Define optimizer
opt = ADAM(0.01) 

# Training loop
for epoch in 1:100

  # Loop over batches
  for (input, target) in train_data

    # Forward pass 
    out = model(input)

    # Compute loss
    loss = postprocess(out, target) 

    # Backprop
    back!(loss)

    # Update model
    update!(opt, params(model))

  end

  # Compute validation loss
  val_loss = evaluate(model, val_input, val_target)
  
  @info "Epoch $epoch, Train loss: $loss, Val loss: $val_loss"

end