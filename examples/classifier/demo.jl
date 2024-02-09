using Flux
using CUDA, cuDNN
using Flux: DataLoader
using DataFrames
using Arrow
using BSON, JSON
using Dates
using DataStructures
using Printf
using ProgressMeter
using Random
using StatsBase
using Unicode

#using TokenizersLite # Uncomment if using bpe or affixer tokenizers below
using TransformersLite
include("../../common/vocab.jl")
include("../../common/training.jl")

## Config
fingerprint = "724e94f4b0c6c405ce7e476a6c5ef4f87db30799ad49f765094cf9770e0f7609"
data_dir = normpath(joinpath(@__DIR__, "..", "..", "datasets\\amazon_reviews_multi\\en\\1.0.0", fingerprint))
vocab_directory = "vocab"
filename = "amazon_reviews_multi-train.arrow"
to_device = gpu # gpu or cpu
target_column = :stars
document_column = :review_body
max_sentence_length = 50

hyperparameters = Dict(
    "seed" => 2718,
    "tokenizer" => "none", # options: none bpe affixes
    "nlabels" => 5,
    "pdrop" => 0.1,
    "dim_embedding" => 32
)
nlabels = hyperparameters["nlabels"]
n_epochs = 10

## Data
filepath = joinpath(data_dir, filename)
df = DataFrame(Arrow.Table(filepath))
display(first(df, 20))
println("")

## Tokenizers

if hyperparameters["tokenizer"] == "bpe"
    path_rules = joinpath(vocab_directory, "bpe", "amazon_reviews_train_en_rules.txt")
    path_vocab = joinpath(vocab_directory, "bpe", "amazon_reviews_train_en_vocab.txt")
    tokenizer = load_bpe(path_rules, startsym="⋅")
elseif hyperparameters["tokenizer"] == "affixes"
    path_vocab = joinpath(vocab_directory, "affixes", "amazon_reviews_train_en_vocab.txt")
    tokenizer = load_affix_tokenizer(path_vocab)
elseif hyperparameters["tokenizer"] == "none"
    path_vocab = joinpath(vocab_directory, "amazon_reviews_train_en.txt")
    tokenizer = identity
end

#vocab = load_vocab(joinpath(@__DIR__, path_vocab))
corpus = String.(df[!, :review_body])
vocab = select_vocabulary(corpus; min_document_frequency=30)
indexer = IndexTokenizer(vocab, "[UNK]")

display(tokenizer)
println("")
display(indexer)
println("")

## Tokens

println("Preparing tokens")
documents = df[!, document_column]
labels = df[!, target_column]
@time tokens = map(d->preprocess(d, tokenizer; max_length=max_sentence_length), documents)
@time indices = indexer(tokens)

y_labels = Int.(labels)
if nlabels == 1
    y_labels[labels .≤ 2] .= 0
    y_labels[labels .≥ 4] .= 1
    idxs = labels .!= 3
    Y = reshape(y_labels, 1, :)
else
    idxs = Base.OneTo(length(labels))
    Y = Flux.onehotbatch(y_labels, 1:nlabels)
end

X_train, Y_train = indices[:, idxs], Y[:, idxs];
rng = MersenneTwister(hyperparameters["seed"])
train_data, val_data = split_validation(rng, X_train, Y_train)

println("train samples:      ", size(train_data[1]), " ", size(train_data[2]))
println("validation samples: ", size(val_data[1]), " ", size(val_data[2]))
println("")

## Model 
dim_embedding = hyperparameters["dim_embedding"]
pdrop = hyperparameters["pdrop"]
model = TransformersLite.TransformerClassifier(
    Embed(dim_embedding, length(indexer)), 
    PositionEncoding(dim_embedding), 
    Dropout(pdrop),
    TransformerEncoderBlock[
        TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop)
    ],
    Dense(dim_embedding, 1), 
    FlattenLayer(),
    Dense(max_sentence_length, nlabels)
    )
display(model)
println("")
model = to_device(model) 

hyperparameters["model"] = "$(typeof(model).name.wrapper)"
hyperparameters["trainable parameters"] = sum(length, Flux.params(model));

if nlabels == 1
    loss(ŷ::AbstractMatrix, y::AbstractMatrix) = Flux.logitbinarycrossentropy(ŷ, y)
    accuracy(ŷ::AbstractMatrix, y::AbstractMatrix) = mean((Flux.sigmoid.(ŷ) .> 0.5) .== y)
else
    loss(ŷ::AbstractMatrix, y::AbstractMatrix) = Flux.logitcrossentropy(ŷ, y)
    accuracy(ŷ::AbstractMatrix, y::AbstractMatrix) = mean(Flux.onecold(ŷ) .== Flux.onecold(y))
end

## Training
batch_size = 32
train_data_loader = DataLoader(train_data |> to_device; batchsize=batch_size, shuffle=true)
val_data_loader = DataLoader(val_data |> to_device; batchsize=batch_size, shuffle=false)

println("Calculating initial metrics")
@time metrics = batched_metrics(model, val_data_loader, accuracy, loss)
@printf "val_acc=%.4f%% ; " metrics.accuracy * 100
@printf "val_loss=%.4f \n" metrics.loss
println("")

output_dir = normpath(joinpath(@__DIR__, "outputs", Dates.format(now(), "yyyymmdd_HHMM")))
mkdir(output_dir)
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
    loss, model, train_data_loader, opt_state, val_data_loader
    ; num_epochs=n_epochs)
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9

## Save 
model = model |> cpu
if hasproperty(tokenizer, :cache)
    # empty cache
    tokenizer = similar(tokenizer)
end
BSON.bson(
    output_path, 
    Dict(
        :model=> model, 
        :tokenizer=>tokenizer,
        :indexer=>indexer
    )
    )
println("saved model to $(output_path).")

open(history_path,"w") do f
  JSON.print(f, history)
end
println("saved history to $(history_path).")
println("")
