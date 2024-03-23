using Flux

function full_loss(Ŷ::AbstractArray{T, 3}, Y::AbstractMatrix{Int}) where T
    # calculate the loss for shifting the whole matrix down by 1
    vocab_size = size(Ŷ, 1) 
    Ŷ = reshape(Ŷ, vocab_size, :) # (vocab_size, N*B)
    Y = Flux.onehotbatch(view(Y, :), 1:vocab_size) # (vocab_size, N*B)
    Flux.logitcrossentropy(Ŷ, Y)
end

function generation_loss(Ŷ::AbstractArray{T, 3}, Y::AbstractMatrix{Int}) where T
    # calculate the loss for only the generation part (last row)
    vocab_size = size(Ŷ, 1) 
    Ygen = Ŷ[:, end, :] # (vocab_size, B)
    Y = Flux.onehotbatch(Y[end, :], 1:vocab_size) # (vocab_size, B)
    Flux.logitcrossentropy(Ygen, Y) 
end

function full_accuracy(Ŷ::AbstractArray{T, 3}, Y::AbstractMatrix{Int}) where T
    # calculate the accuracy for shifting the whole matrix down by 1
    vocab_size = size(Ŷ, 1) 
    Ŷ = reshape(Ŷ, vocab_size, :) # (vocab_size, N*B)
    ŷ = Flux.onecold(Ŷ) # (N*B,)
    y = view(Y, :) # (N*B,)
    mean(ŷ .== y)
end

function generation_accuracy(Ŷ::AbstractArray{T, 3}, Y::AbstractMatrix{Int}) where T
    # calculate the accuracy for only the generation part (last row)
    ygen = Flux.onecold(Ŷ[:, end, :]) # (B,)
    y = Y[end, :] # (B,)
    mean(ygen .== y) 
end