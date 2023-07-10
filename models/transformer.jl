module Models

using ..Attention
using ..TransformerBlocks

"""
Transformer model. Consists of an encoder, decoder and RL agent.
"""
struct Transformer
    encoder::TransformerEncoder
    decoder::TransformerDecoder
    rl_agent::RLAgent  # Abstract type, implementation details hidden
end

function Transformer(; n_layers=6, n_heads=8, dim=512, dim_ff=2048, 
                    max_len=512, src_vocab=40000, trg_vocab=40000,  
                    rl_agent)
    attn = Attention(dim, n_heads) 
    ff = FeedForward(dim, dim_ff)
    position = PostionEmbedding(dim, max_len)
    
    encoder = TransformerEncoder(n_layers, attn, ff, position, src_vocab)
    decoder = TransformerDecoder(n_layers, attn, ff, position, trg_vocab)
    
    return Transformer(encoder, decoder, rl_agent) 
end

function forward(model::Transformer, src, trg, mode) 
    # Pass the source sequence through the encoder
    encoder_output = forward(model.encoder, src)

    # Initialize the decoder input as the start token
    decoder_input = trg[:, 1]

    # Initialize an empty tensor to hold the decoder outputs
    decoder_outputs = []

    # Depending on the mode, we may use teacher forcing or not
    if mode == "teacher_forcing"
        for t in 2:size(trg, 2)
            # Pass the encoder output and the decoder input through the decoder
            decoder_output, decoder_state = forward(model.decoder, decoder_input, encoder_output)
            push!(decoder_outputs, decoder_output)

            # Use the actual next token as the next input to the decoder
            decoder_input = trg[:, t]
        end
    elseif mode == "feedback"
        for t in 2:size(trg, 2)
            # Pass the encoder output and the decoder input through the decoder
            decoder_output, decoder_state = forward(model.decoder, decoder_input, encoder_output)
            push!(decoder_outputs, decoder_output)

            # Use the decoder output as the next input to the decoder
            decoder_input = argmax(decoder_output, dims=2)
        end
    else
        error("Invalid mode: $mode")
    end

    # Pass the decoder outputs through the RL agent
    if model.rl_agent isa UniformRLAgent
        rl_output = uniform_sample(model.rl_agent)
    elseif model.rl_agent isa ReinforceRLAgent
        rl_output = reinforce_sample(model.rl_agent, decoder_outputs)
    else
        error("Invalid RL agent: $(typeof(model.rl_agent))")
    end
    return rl_output
end

end # module