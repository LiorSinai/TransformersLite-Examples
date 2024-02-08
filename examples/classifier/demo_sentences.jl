using DataFrames
using Arrow
using Printf
using BSON, JSON
using Flux
using Flux: DataLoader
using Unicode
using Dates
using StatsBase: mean

using TokenizersLite
using TransformersLite
include("../../common/vocab.jl")
include("../../common/training.jl")
include("../../common/SentenceClassifier.jl")

## Config
fingerprint = "724e94f4b0c6c405ce7e476a6c5ef4f87db30799ad49f765094cf9770e0f7609"
data_dir = normpath(joinpath(@__DIR__, "..", "datasets\\amazon_reviews_multi\\en\\1.0.0", fingerprint))
filename = "amazon_reviews_multi-train.arrow"
to_device = gpu # gpu or cpu
target_column = :stars
document_column = :review_body

hyperparameters = Dict(
    "seed" => 2718,
    "tokenizer" => "sentences+bpe",
    "nlabels" => 1,
    "pdrop" => 0.1,
    "dim_embedding" => 32,
    "max_sentence_length" => 30,
)
nlabels = hyperparameters["nlabels"]
max_sentence_length = hyperparameters["max_sentence_length"]
num_epochs = 10

## Data
filepath = joinpath(data_dir, filename)
df = DataFrame(Arrow.Table(filepath))
display(first(df, 10))
println("")

## Tokenizers
sentence_splitter = RuleBasedSentenceSplitter()

output_dir = joinpath(@__DIR__, "..", "vocab\\bpe")
path_rules = joinpath(output_dir, "amazon_reviews_train_en_rules.txt")
path_vocab = joinpath(output_dir, "amazon_reviews_train_en_vocab.txt")
tokenizer = load_bpe(path_rules, startsym="⋅")

vocab = load_vocab(path_vocab)
indexer = IndexTokenizer(vocab, "[UNK]")

display(sentence_splitter)
display(tokenizer)
display(indexer)
println("")

## Tokens
println("Preparing tokens")

function pad!(v::Vector{String}, symbol::String, max_length::Int)
    if length(v) < max_length
        padding = fill(symbol, max_length - length(v))
        append!(v, padding)
    end
end

documents = df[!, document_column]
labels = df[!, target_column]

# Set polarity
y_train = copy(labels)
y_train[labels .≤ 2] .= 0
y_train[labels .≥ 4] .= 1
idxs = labels .!= 3
y_train = y_train[idxs]

tokens = Vector{Vector{String}}[]
@time for doc in documents[idxs]
    sentences = sentence_splitter(doc)
    tokens_doc = map(s->preprocess(s, tokenizer; max_length=max_sentence_length), sentences)
    pad!(tokens_doc[1], tokenizer.unksym, max_sentence_length) # hack to ensure all indices have common length
    push!(tokens, tokens_doc)
end
@time indices = map(indexer, tokens) 

rng = MersenneTwister(hyperparameters["seed"])
train_data, val_data = split_validation(rng, indices, y_train)

println("train samples:      ", size(train_data[1]), " ", size(train_data[2]))
println("validation samples: ", size(val_data[1]), " ", size(val_data[2]))
println("")

## Model 
dim_embedding = hyperparameters["dim_embedding"]
pdrop = hyperparameters["pdrop"]

base_model = TransformersLite.TransformerClassifier(
    Embed(dim_embedding, length(indexer)), 
    PositionEncoding(dim_embedding), 
    Dropout(pdrop),
    TransformerEncoderBlock[TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop)],
    Dense(dim_embedding, 1), 
    FlattenLayer(),
    Dense(max_sentence_length, nlabels)
    )
display(base_model)

model = SentenceClassifier(
    base_model,
    Flux.sigmoid,
    parabolic_weighted_average
)
display(model)
println("")

hyperparameters["model"] = "$(typeof(model).name.wrapper)-$(typeof(model.base_model).name.wrapper)"
hyperparameters["trainable parameters"] = sum(length, Flux.params(model));

loss(ŷ::AbstractVector, y::AbstractVector) = Flux.binarycrossentropy(ŷ, y)
accuracy(ŷ::AbstractVector, y::AbstractVector) = mean((ŷ .> 0.5) .== y)

## Training 
batch_size = 32
train_data_loader = DataLoader(train_data; batchsize=batch_size, shuffle=true)
val_data_loader = DataLoader(val_data; batchsize=batch_size, shuffle=false)

println("Calculating initial metrics")
@time metrics = batched_metrics(model, val_data_loader, loss, accuracy)
@printf "val_acc=%.4f%%; " metrics.accuracy * 100
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
    ; num_epochs=num_epochs
    )
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9

## Save 

if hasproperty(tokenizer, :cache)
    tokenizer = similar(tokenizer)
end
BSON.bson(
    output_path, 
    Dict(
        :model=> model, 
        :tokenizer=>tokenizer,
        :indexer=>indexer,
        :sentence_splitter=>sentence_splitter
    )
    )
println("saved model to $(output_path).")

open(history_path,"w") do f
  JSON.print(f, history)
end
println("saved history to $(history_path).")