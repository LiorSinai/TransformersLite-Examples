using Flux
using CUDA, cuDNN
using BSON, JSON
using Printf
using Dates
using StatsBase

using TransformersLite
using TransformersLite: decode
using TransformersLite: make_causal_mask
include("../../common/training.jl")
include("loss.jl")
include("generate_batches.jl")

## Config
frac_validation = 0.05
context_size = 32
batch_size = 32
# There are 4.96 million characters, of which 95% are used for training
# Sampling randomly, indices will repeat approximately every 
#   (4.71 million) / (batches_per_epoch * batch_size * context_size) steps.
# and each index will be first approximately every 
#   (4.71 million) / (batches_per_epoch * batch_size) steps.
batches_per_epoch = 4600
n_epochs = 10
data_path = normpath(joinpath(@__DIR__, "..", "..", "datasets", "shakespeare_plays.txt"))
output_dir = normpath(joinpath(@__DIR__, "outputs", Dates.format(now(), "yyyymmdd_HHMM")))
to_device = gpu # gpu or cpu

hyperparameters = Dict{String, Any}(
    "seed" => 2718,
    "pdrop" => 0.1,
    "dim_embedding" => 32,
    "context_size" => context_size,
)

mkdir(output_dir)

## Data

println("Loading data")
text = open(data_path) do file
    read(file, String)
end

println(text[1:500], "\n...")
println("")
println("lines:      ", count('\n', text))
println("characters: ", length(text))
println("words:      ", count(r"\w+", text))
println("")

## Tokens

println("Preparing tokens")
characters = sort(collect(Set(text)))
vocab_size = length(characters)
println(length(characters), ": ", replace(join(characters), '\n'=>"\\n"))

push!(characters, 'Ø')
indexer = IndexTokenizer(characters, 'Ø')
println(indexer)

println(indexer(collect("hii there")))
println(join(decode(indexer, indexer(collect("hii there")))))
println("")

tokens = indexer(collect(text))
n_val = floor(Int, (1 - frac_validation) * length(tokens))
train_data = tokens[1:n_val]
val_data = tokens[(n_val + 1):end]

println("train tokens:      ", length(train_data))
println("validation tokens: ", length(val_data))
println("")

## Batches
### Train on shifted outputs. 
### Each input column tries to predict a new column shifted by one, including generation of a new last token

batch_generator = BatchGenerator(
    to_device(train_data);
    context_size=context_size, 
    batch_size=batch_size,
    num_steps=batches_per_epoch * batch_size,
    )
val_generator = BatchGenerator(
    to_device(val_data);
    context_size=context_size,
    batch_size=batch_size
)

println("batch sizes: ", size.(first(batch_generator)))

## Model
dim_embedding = hyperparameters["dim_embedding"]
pdrop = hyperparameters["pdrop"]
mask = make_causal_mask(ones(context_size, context_size))

model = TransformersLite.TransformerGenerator(
    Embed(dim_embedding, vocab_size),
    PositionEncoding(dim_embedding, context_size), 
    Dropout(0.1),
    TransformerBlock[
        TransformerBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop),
        TransformerBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop),
        TransformerBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop),
    ],
    Dense(dim_embedding, vocab_size),
    copy(mask)
    )
display(model)
println("")
model = to_device(model)

hyperparameters["model"] = "$(typeof(model).name.wrapper)"
hyperparameters["trainable parameters"] = sum(length, Flux.params(model));

Xb, Yb = first(batch_generator)
println("sizes Xb, Yb: ", size(Xb), " ", size(Yb))
Y_est = model(Xb)
println("size model(Xb): ", size(Y_est))
println("")

println("initial generation:")
context = reshape([1], 1, 1)
context = generate(model, context; context_size=context_size, max_tokens=500)
decoded_context = decode(indexer, context[:, 1])
println(join(decoded_context))
println("")

loss = full_loss
accuracy = full_accuracy

## Training

println("Calculating initial metrics")
@printf "expected accuracy: %.4f%%; " 1/vocab_size * 100
@printf "expected loss: %.4f\n" -log(1/vocab_size) 
metrics = batched_metrics(model, val_generator, loss, accuracy)
@printf "val_acc=%.4f%% ; " metrics.full_accuracy * 100
@printf "val_loss=%.4f \n" metrics.full_loss
println("")

output_path = joinpath(output_dir, "model.bson")
history_path = joinpath(output_dir, "history.json")
hyperparameter_path = joinpath(output_dir, "hyperparameters.json")

open(hyperparameter_path, "w") do f
    JSON.print(f, hyperparameters)
end
println("saved hyperparameters to $(hyperparameter_path).")
println("")

println("Training ...")
opt_state = Flux.setup(Adam(), model)
start_time = time_ns()
history = train!(
    loss, model, batch_generator, opt_state, val_generator
    ; num_epochs=n_epochs)
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9

println("after training generation:")
context = reshape([1], 1, 1)
context = generate(model, context; context_size=context_size, max_tokens=500)
decoded_context = decode(indexer, context[:, 1])
println(join(decoded_context))
println("")

## save
model = model |> cpu
BSON.bson(
    output_path, 
    Dict(
        :model=> model, 
        :indexer => indexer,
    )
    )
println("saved model to $(output_path).")

open(history_path,"w") do f
    JSON.print(f, history)
end
println("saved history to $(history_path).")
println("")
