function evaluate(model, data)
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    for batch in data
        preds = model(batch) 
        loss = loss_fn(preds, batch["labels"])
        total_loss += loss
        
        # Calculate accuracy
        max_preds = argmax(preds, dims=2) .== batch["labels"]
        total_acc += mean(max_preds) * size(batch["labels"])[1]  # Multiply by seq_len 
        
        n_batches += 1
    end
    
    avg_loss = total_loss / n_batches
    avg_acc = total_acc / size(data["labels"])[2]  # Divide by total number of labels
    return avg_loss, avg_acc
end