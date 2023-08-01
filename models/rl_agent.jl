module RLAgent 

"""
Abstract RL agent type
"""
# TODO: finishing the RLAgent module
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
    policy_net::Net  # Neural network 
    value_net::Net   # Neural network
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


function update_reinforce!(agent::ReinforceRLAgent, rewards) 
    ...
end 

end # module