using JSON 
using CUDA

# Load config
config = JSON.parsefile("configs/config.json")

# Set device 
device = CUDA.has_cuda() ? "cuda" : "cpu"

# Load data
data = load("data/data.jl")

# Load model 
model = if config["model_type"] == "transformer" 
    TransformerModel(config["embed_size"], config["num_heads"], config["feed_forward_size"]) 
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
    for batch in data 
        # Zero gradients
        Optimizer.zero_grad!(optimizer) 
        
        # Get predictions 
        preds = model(batch) 
        
        # Calculate loss
        loss = loss_fn(preds, batch["labels"])
        
        # Backpropagate 
        backprop!(loss, model)
        
        # Update weights 
        update!(optimizer, model)
    end
    
    # Evalute on validation set
    val_loss = evaluate(model, val_data)
    
    # Log results
    @info "Epoch $epoch | Train loss: $loss | Val loss: $val_loss" 
end