module RLAgent 

using StringDistances

"""
Abstract RL agent type
"""
abstract type RLAgent end

""" 
Vanilla RL agent that samples from a uniform distribution over target vocab.
"""
struct UniformRLAgent <: RLAgent
    trg_vocab::Int 
end

function uniform_sample(agent::UniformRLAgent)
    return rand(1:agent.trg_vocab) 
end

""" 
Reinforce RL agent that maintains a policy and value network.
""" 
struct ReinforceRLAgent <: RLAgent
    policy_net::Net   
    value_net::Net   
    optimizer::Optimizer
    encoder::TransformerEncoder
    decoder::TransformerDecoder
end

function reinforce_sample(agent::ReinforceRLAgent, encoder_output, decoder_output)
    # Concatenate encoder and decoder outputs
    state = cat(encoder_output, decoder_output, dims=2)

    # Pass state through policy network to get action probabilities
    probs = softmax(agent.policy_net(state))

    # Sample an action from the action probabilities
    action = sample(probs)

    return action
end

function update_reinforce!(agent::ReinforceRLAgent, states, actions, targets)

    # Calculate the errors made by the transformer model
    errors = calculate_errors(actions, targets) 

    # Calculate rewards based on the errors
    rewards = calculate_rewards(errors) 

    # Calculate discounted rewards
    discounts = discount(rewards, 0.99)  

    # Get log probabilities of actions taken 
    log_probs = log.(agent.policy_net(states))

    # Calculate loss  
    loss = -sum(discounts .* log_probs)

    # Backpropagate loss
    back!(loss)

    # Update policy network
    update!(agent.optimizer, params(agent.policy_net))

    # Update value network
    values = agent.value_net(states)
    value_loss = mse(values, discounts)

    back!(value_loss)
    update!(agent.optimizer, params(agent.value_net))

end

# Define his function to calculate errors (sequence-to-sequence task)
function calculate_errors(actions, targets)
    return [levenshtein(action, target) for (action, target) in zip(actions, targets)]
end

# Define this function to calculate rewards (normalize the rewards to ensure that they are on a consistent scale across different examples and batches)
function calculate_rewards(errors)
    max_error = maximum(errors)
    min_error = minimum(errors)
    return 1 .- (errors .- min_error) ./ (max_error - min_error)
end

end # module