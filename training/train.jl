include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/attention/attention.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/models/transformer.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/models/rl_agent.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/data.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/embeddings/embeddings.jl")

using JSON 
using CUDA

# Load config
config = JSON.parsefile("configs/config.json")

# Set device 
device = CUDA.has_cuda() ? "cuda" : "cpu"

# Define paths and max_len
src_path = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/source.txt"
trg_path = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/target.txt"
max_len = 100 

# Load and preprocess data
data = process_data(src_path, trg_path, max_len, tokenizer=default_tokenizer)

# Load model 
model = if config["model_type"] == "transformer" 
    Transformer(n_layers=config["n_layers"], n_heads=config["n_heads"], dim=config["dim"], dim_ff=config["dim_ff"], max_len=config["max_len"], src_vocab=config["src_vocab"], trg_vocab=config["trg_vocab"], rl_agent=config["rl_agent"])
elseif config["model_type"] == "rl_agent"
    RLAgentModel(config["state_size"], config["action_size"])
end

# Move model to device
model = model |> device 

# Define optimizer 
optimizer = Adam(config["lr"])

# Define loss function 
loss_fn = CrossEntropy() 

# Train loop
for epoch in 1:config["epochs"]
    for (src_batch, trg_batch) in data
        # Preprocess the batch
        src_seq, src_pad_mask, trg_input, trg_pad_mask, trg_target = preprocess(src_batch, trg_batch)

        # Move data to device
        src_seq = src_seq |> device
        src_pad_mask = src_pad_mask |> device
        trg_input = trg_input |> device
        trg_pad_mask = trg_pad_mask |> device
        trg_target = trg_target |> device

        # Zero gradients
        Optimizer.zero_grad!(optimizer)

        # Get predictions
        preds = model(src_seq, src_pad_mask, trg_input, trg_pad_mask)

        # Calculate loss
        loss = postprocess(preds, trg_target)

        # Backpropagate
        backprop!(loss, model)

        # Update weights
        update!(optimizer, model)
    end

    # Evaluate on validation set (you'll need to define this part)
    val_loss = evaluate(model, val_data)

    # Log results
    @info "Epoch $epoch | Train loss: $loss | Val loss: $val_loss"
end
