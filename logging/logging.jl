module LoggingModule

export log_attention_weights, log_encoder_decoder_output, log_training_metrics, log_tensor_shapes, log_loss_value

# Check for NaN values in an array
function check_for_nan(array, name)
    if any(isnan, array)
        println("$name contains NaN values.")
    end
end

# Log attention weights
function log_attention_weights(weights)
    println("Attention weights: $weights")
    check_for_nan(weights, "Attention Weights")
end

# Log encoder and decoder outputs
function log_encoder_decoder_output(encoder_output, decoder_output)
    println("Encoder Output: $encoder_output")
    println("Decoder Output: $decoder_output")
    check_for_nan(encoder_output, "Encoder Output")
    check_for_nan(decoder_output, "Decoder Output")
end

# Log gradients or any other training metrics
function log_training_metrics(metrics)
    println("Training Metrics: $metrics")
    # Add more specific checks here
end

# Log tensor shapes
function log_tensor_shapes(src_seq, trg_input, preds)
    println("Shape of src_seq: ", size(src_seq))
    println("Shape of trg_input: ", size(trg_input))
    println("Shape of preds: ", size(preds))
end

# Log loss value
function log_loss_value(loss)
    println("Calculated loss: ", loss)
end

end # module
