include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/attention/attention.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/models/transformer.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/RLAgentModule/rl_agent.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/data.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/embeddings/embeddings.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/evaluation/evaluation.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/logging/logging.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/pipelines/preprocessing.jl")
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/pipelines/ingestion.jl")

using JSON 
using CUDA
using Flux
using .LoggingModule

# Load config
config = JSON.parsefile("configs/config.json")

# Set device 
device = CUDA.has_cuda() ? "cuda" : "cpu"

# Define paths and max_len
src_path = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/source.txt"
trg_path = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/target.txt"
max_len = 100 
tokenizer = Ingestion.load_data("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/vocab.json")

# Load and preprocess data
data = Preprocessing.process_data(src_path, trg_path, max_len, tokenizer)

# Load model (cf. config.json file for the hyperparameters)
attention = Attention(config["attention_dim"], config["attention_heads"])
layer1_size = (2048, 512)
layer2_size = (512, 128)

feed_forward_layer1 = rand(Float64, layer1_size...)
feed_forward_layer2 = rand(Float64, layer2_size...)

feed_forward = FeedForward(feed_forward_layer1, feed_forward_layer2)

function create_position_encoding(encoding_type::String, max_len::Int, dim::Int)
    if encoding_type == "sine"
        # Create sine encoding
        return sine_encoding(max_len, dim)
    else
        # Handle other encoding types or throw an error
        error("Unknown position encoding type: $encoding_type")
    end
end

function sine_encoding(max_len::Int, dim::Int)
    pos = 0:max_len-1
    div_term = exp.(2 .* log.(10.0) .* -(0:2:dim-2) ./ dim)
    pos_embedding = zeros(Float64, max_len, dim)
    pos_embedding[:, 1:2:dim-1] .= sin.(pos .* div_term')
    pos_embedding[:, 2:2:dim] .= cos.(pos .* div_term')
    return pos_embedding
end

position_encoding_matrix = create_position_encoding(config["position_encoding"], max_len, config["dim"])
position_encoding = PositionEncoding(position_encoding_matrix)

encoder = TransformerEncoder(config["n_layers"], attention, feed_forward, position_encoding)
decoder = TransformerDecoder(config["n_layers"], attention, feed_forward, position_encoding, attention)

function create_layers(layer_sizes::Vector{Int})
    layers = Layer[]
    for i in 1:length(layer_sizes) - 1
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        weights = rand(Float64, output_size, input_size)
        biases = rand(Float64, output_size)
        layer = Layer(weights, biases, relu)
        push!(layers, layer)
    end
    return layers
end

policy_layers = create_layers(Vector{Int}(config["policy_layers"]))
value_layers = create_layers(Vector{Int}(config["value_layers"]))

policy_net = Net(policy_layers)
value_net = Net(value_layers)

optimizer = SGDOptimizer(config["learning_rate"])

rl_agent = ReinforceRLAgent(policy_net, value_net, optimizer, encoder, decoder)

model = Transformer(encoder, decoder, rl_agent)

# Move model to device
if CUDA.has_cuda()
    model = gpu(model)
end

# Define optimizer 
optimizer = Flux.Adam(config["learning_rate"])

# Define loss function 
loss_fn = Flux.Losses.crossentropy

# Validation data
val_src_path = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/source.txt"
val_trg_path = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/data/target.txt"
val_data = Preprocessing.process_data(val_src_path, val_trg_path, max_len, tokenizer)

# Train loop
for epoch in 1:config["epochs"]
    println("Starting epoch $epoch")  # Debug log
    println("Size of data: ", length(data))
    println("First element of data: ", data[1])
    println("Second element of data: ", data[2])
    println("Size of first element of data: ", length(data[1]))
    println("Size of second element of data: ", length(data[2]))
    if isempty(data[1]) || isempty(data[2])
        error("Data is empty. Cannot proceed with training.")
    end    

    total_train_loss = 0.0
    n_batches = 0

    if isempty(data)
        println("Data is empty. Exiting.")
        break
    end
    
    for (src_batch, trg_batch) in data
        
        println("Processing a new batch")
        println("Size of src_batch: ", length(src_batch))
        println("Size of trg_batch: ", length(trg_batch))
        
        # Preprocess the batch
        src_seq, src_pad_mask, trg_input, trg_pad_mask, trg_target = preprocess(src_batch, trg_batch)

        if isempty(src_seq) || isempty(trg_input) || isempty(trg_target)
            println("Preprocessed tensors are empty. Skipping this batch.")
            continue
        end

        # Log tensor shapes
        log_tensor_shapes(src_seq, trg_input, preds)

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

        # Log tensor shapes after prediction
        log_tensor_shapes(src_seq, trg_input, preds)

        # Calculate loss
        loss = postprocess(preds, trg_target)
        total_train_loss += loss

        # Backpropagate
        backprop!(loss, model)

        # Log loss value
        log_loss_value(loss)

        # Update weights
        update!(optimizer, model)

        n_batches += 1
    end

    avg_train_loss = total_train_loss / n_batches

    # Evaluate on validation set
    val_loss = evaluate(model, val_data)

    # Log results
    @info "Epoch $epoch | Train loss: $avg_train_loss | Val loss: $val_loss"
end
