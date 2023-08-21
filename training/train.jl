include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/attention/attention.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/models/transformer.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/RLAgentModule/rl_agent.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/data.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/embeddings/embeddings.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")

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

# Load model (cf. config.json file for the hyperparameters)
attention = Attention(config["attention_dim"], config["attention_heads"])
layer1_size = (2048, 512)
layer2_size = (512, 128)

feed_forward_layer1 = rand(Float64, layer1_size...)
feed_forward_layer2 = rand(Float64, layer2_size...)

feed_forward = FeedForward(feed_forward_layer1, feed_forward_layer2)
position_encoding = PositionEncoding(encoding=config["position_encoding"])

encoder = TransformerEncoder(n_layers=config["encoder_layers"], attention=attention, feed_forward=feed_forward, position_encoding=position_encoding)
decoder = TransformerDecoder(n_layers=config["decoder_layers"], attention=attention, feed_forward=feed_forward, position_encoding=position_encoding, encoder_attention=attention)

policy_net = Net(layers=config["policy_layers"])
value_net = Net(layers=config["value_layers"])
optimizer = SGDOptimizer(learning_rate=config["learning_rate"])

rl_agent = ReinforceRLAgent(policy_net=policy_net, value_net=value_net, optimizer=optimizer, encoder=encoder, decoder=decoder)

model = Transformer(encoder=encoder, decoder=decoder, rl_agent=rl_agent)

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
