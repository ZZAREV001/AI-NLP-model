module Models

using ..Attention
using ..TransformerBlocks

"""
Transformer model. Consists of an encoder, decoder and RL agent.
"""
struct Transformer
    encoder::TransformerEncoder
    decoder::TransformerDecoder
    rl_agent::RLAgent 
end
  
function Transformer(; n_layers, n_heads, dim, dim_ff, max_len, src_vocab, trg_vocab, rl_agent)
  
    # Attention uses linear attention 
    attn = LinearAttention(dim, n_heads)
    
    # TransformerBlock has token shift
    ff = TransformerBlock(dim, dim_ff) 
    
    position = PositionalEncoding(dim, max_len)
  
    # Encoder and decoder handle receptance
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