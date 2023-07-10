# Extract text from PDFs
text = ""
for pdf in pdf_files
    text += PDFPlumber.extract_text(pdf) 
end

# Preprocess text
text = lowercase(text)
text = remove_punctuation(text) 
# ...

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

# Train loop 
for epoch in 1:100
    for batch in DataLoader(input_tensor, target_tensor, batch_size=64)
        # Train model on batch
        # ...
    end
    
    # Evaluate on val 
    val_loss = evaluate(model, val_input, val_target)
end  

# Test on test set
test_loss, test_acc = evaluate(model, test_input, test_target)

@info "Test loss: $test_loss, Test acc: $test_acc"