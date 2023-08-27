using Flux
using Test
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/attention/attention.jl")

@testset "Linear Attention" begin

    # Inputs
    query = randn(2, 4, 8) # (Batch, SeqLen, Dim)
    key = randn(2, 6, 8)
    value = randn(2, 6, 16)
  
    attn_module = linear_attention(8, 2)
  
    # Compute attention 
    attn_out = linear_attention(query, attn_module)
  
    # Test shape
    @test size(attn_out) == (2, 4, 16)
  
    # Test scores sum to 1
    scores = attn_module(query, key)
    @test ≈(sum(scores, dims=2), 1.0)
  
    # Test mask    
    mask = ones(2, 6) 
    mask[1, 3:6] .= 0 # Mask out
  
    out = linear_attention(query, attn_module; mask)
  
    @test out[1, 3:4, :] .≈ 0 # Attended locations are masked
  
    # Test passing value
    out = linear_attention(query, key, value, attn_module)
  
    @test size(out) == size(value) # Outputs match value sizes
  
  end