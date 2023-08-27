module Models

using .LoggingModule
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/common-definition/common.jl")

"""
Transformer model. Consists of an encoder, decoder and RL agent.
"""

function Transformer(; n_layers, n_heads, dim, dim_ff, max_len, src_vocab, trg_vocab, rl_agent)
  
    # Attention uses linear attention 
    attn = LinearAttention(dim, n_heads)
    
    # TransformerBlock has token shift
    ff = TransformerBlock(dim, dim_ff) 
    
    position = PositionalEncoding(dim, max_len)
  
    # Encoder and decoder handle receptance
    encoder = TransformerEncoder(n_layers, attn, ff, position, src_vocab) 
    decoder = TransformerDecoder(n_layers, attn, ff, position, trg_vocab)

    # Call logging functions
    log_encoder_decoder_output(encoder_output, decoder_output)
  
    return Transformer(encoder, decoder, rl_agent)
end

function forward(model::Transformer, src, trg, mode) 
    # Pass the source sequence through the encoder
    encoder_output = forward(model.encoder, src)

    # Initialize the decoder input as the start token
    decoder_input = trg[:, 1]

    # Initialize an empty tensor to hold the decoder outputs
    decoder_outputs = []

    # Initialize an empty tensor to hold the attention weights
    attention_weights = []

    # Depending on the mode, we may use teacher forcing or not
    if mode == "teacher_forcing"
        for t in 2:size(trg, 2)
            # Pass the encoder output and the decoder input through the decoder
            decoder_output, decoder_state, attn_weights = forward(model.decoder, decoder_input, encoder_output)
            push!(decoder_outputs, decoder_output)
            push!(attention_weights, attn_weights)

            # Use the actual next token as the next input to the decoder
            decoder_input = trg[:, t]
        end
    elseif mode == "feedback"
        for t in 2:size(trg, 2)
            # Pass the encoder output and the decoder input through the decoder
            decoder_output, decoder_state, attn_weights = forward(model.decoder, decoder_input, encoder_output)
            push!(decoder_outputs, decoder_output)
            push!(attention_weights, attn_weights)

            # Use the decoder output as the next input to the decoder
            decoder_input = argmax(decoder_output, dims=2)
        end
    else
        error("Invalid mode: $mode")
    end

    # Pass the decoder outputs and attention weights through the RL agent
    if model.rl_agent isa UniformRLAgent
        rl_output = uniform_sample(model.rl_agent)
    elseif model.rl_agent isa ReinforceRLAgent
        rl_output = reinforce_sample(model.rl_agent, decoder_outputs, attention_weights)
    else
        error("Invalid RL agent: $(typeof(model.rl_agent))")
    end
    return rl_output
end

end # module