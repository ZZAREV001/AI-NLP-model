using Flux
using LinearAlgebra
using Test
include("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/attention/attention.jl")

@testset "Linear Attention" begin

  # Define the input tensors
  k = rand(Float32, (10, 5, 16))
  w = rand(Float32, (5, 8))
  
  # Create a LinearAttention object
  attn = LinearAttention(16, 8)
  
  # Compute the attention weights
  probs = linear_attention(k, w; attn)
  
  # Print the attention weights
  println("Attention weights:")
  println(probs)
  
  end