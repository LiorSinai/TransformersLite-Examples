using Flux
using Flux: DataLoader
using Flux.MLUtils: getobs, numobs
using Random
using ProgressMeter
using Printf

"""
    batched_metrics(model, data, metrics::NamedTuple)
    batched_metrics(model, data, funcs...)

Calculates `f(model(x), y)` for each `(x, y)` in `data` and each `f` in `metrics`, and returns
the average weighted by batch size as a `NamedTuple`.
If `f` takes the mean over a batch this recovers the full sample mean, while only holding one batch in memory.
To automatically batch data, use `Flux.DataLoader`.

With the `NamedTuple` form the keys of the result are the keys of `metrics`.
With the positional form they are the function names.
"""
function batched_metrics(model, data, metrics::NamedTuple)
    results = zeros(Float64, length(metrics))
    num_observations = 0
    @showprogress desc="batch metrics..." for (x, y) in data
        y_model = model(x)
        batch_values = map(f -> f(y_model, y), values(metrics))
        batch_size = count_observations(x)
        results .+= batch_values .* batch_size
        num_observations += batch_size
    end
    results ./= num_observations
    NamedTuple{keys(metrics)}(Tuple(results))
end

batched_metrics(model, data, funcs...) = batched_metrics(model, data, NamedTuple{Symbol.(funcs)}(funcs))

count_observations(data::D) where {D<:DataLoader} = count_observations(data.data)
count_observations(data::Tuple) = count_observations(data[1]) # assume data[1] are samples and data[2] are labels
count_observations(data::AbstractArray{<:Any,N}) where {N} = size(data, N)
count_observations(data) = length(data)

"""
    split_validation(rng, X, Y; frac=0.1)

Splits a data set into a training and validation data set, where observations are along the last dimension.
`frac` is the fraction of observations assigned to the validation set.
Returns `(X_train, Y_train), (X_val, Y_val)`.
"""
function split_validation(rng::AbstractRNG, data, labels; frac::Real=0.1)
    0 <= frac < 1 || throw(ArgumentError("frac must be in [0, 1), got $frac"))
    nsamples = numobs((data, labels)) # throws if data and labels have different numbers of observations
    idxs = randperm(rng, nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    idxs_train = idxs[1:ntrain]
    idxs_val = idxs[(ntrain + 1):end]
    getobs((data, labels), idxs_train), getobs((data, labels), idxs_val)
end

"""
    train!(loss, model, train_data, opt_state, val_data[, history];
        num_epochs=10, metrics=(; loss))

Returns a dictionary `history` of metrics calculated during the train loop.

Modelled after `Flux.train!`. Trains `model` with `loss(model(x), y)` for each `(x, y)` in `train_data`.

After each epoch, each metric in `metrics` (a `NamedTuple` of functions `f(ŷ, y)`) is calculated over
`val_data` and all of `train_data`.

`history["mean_batch_loss"]` is the mean of the batch losses during the epoch.
"""
function train!(loss, model, train_data, opt_state, val_data; kwargs...)
    history = Dict{String,Vector{Float64}}()
    train!(loss, model, train_data, opt_state, val_data, history; kwargs...)
end

function train!(
    loss, model, train_data, opt_state, val_data, history::AbstractDict;
    num_epochs::Int=10, metrics::NamedTuple=(; loss)
    )
    for epoch in 1:num_epochs
        flush(stdout)
        progress = Progress(length(train_data); desc="epoch $epoch/$num_epochs")
        total_loss = 0.0
        num_batches = 0
        for (x, y) in train_data
            batch_loss, grads = Flux.withgradient(model) do m
                loss(m(x), y)
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += batch_loss
            num_batches += 1
            ProgressMeter.next!(
                progress;
                showvalues = () -> [(:mean_loss, total_loss / num_batches), (:batch_loss, batch_loss)]
            )
        end
        ProgressMeter.finish!(progress)
        push!(get!(history, "mean_batch_loss", Float64[]), total_loss / num_batches)
        update_history!(history, model, train_data, "train_", metrics)
        update_history!(history, model, val_data, "val_", metrics)
    end
    println("")
    history
end

function update_history!(history::AbstractDict, model, data, prefix::String, metrics::NamedTuple)
    results = batched_metrics(model, data, metrics)
    for (name, value) in pairs(results)
        metric_name = prefix * String(name)
        push!(get!(history, metric_name, Float64[]), value)
        @printf "%s=%.4f; " metric_name value
    end
    println("")
end
