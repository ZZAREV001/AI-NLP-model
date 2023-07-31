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
    ...
end

function reinforce_sample(agent::ReinforceRLAgent, state)
    probs = agent.policy_net(state) 
    sample = sample(1:size(probs,2), Weights(probs))
    return sample 
end

function update_reinforce!(agent::ReinforceRLAgent, rewards) 
    ...
end 

end # module