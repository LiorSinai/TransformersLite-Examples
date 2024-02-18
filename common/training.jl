using Flux
using Flux: DataLoader
using Random
using ProgressMeter
using Printf

"""
    batched_metrics(model, data, funcs...)

Caculates `f(model(x), y)` for each `(x, y)` in data and each `f` in funcs, and returns a weighted sum by batch size.
If `f` takes the mean this will recover the full sample mean.
Reduces memory load for `f` and `g`. 
To automatically batch data, use `Flux.DataLoader`.
"""
function batched_metrics(model, data, funcs...)
    results = zeros(Float32, length(funcs))
    num_observations = 0
    @showprogress desc="batch metrics..." for (x, y) in data
        y_model = model(x)
        values = map(f->f(y_model, y), funcs)
        batch_size = count_observations(x) 
        results .+= values .* batch_size
        num_observations += batch_size
    end
    results /= num_observations
    (; zip(Symbol.(funcs), results)...)
end

count_observations(data::D) where {D<:DataLoader} = count_observations(data.data)
count_observations(data::Tuple) = count_observations(data[1]) # assume data[1] are samples and data[2] are labels
count_observations(data::AbstractArray{<:Any,N}) where {N} = size(data, N)
count_observations(data) = length(data)

"""
    split_validation(rng, X, Y; frac=0.1)

Splits a data set into a training and validation data set.
"""
function split_validation(rng::AbstractRNG, data::AbstractArray, labels::AbstractVecOrMat; frac::Float64=0.1)
    nsamples = size(data)[end]
    idxs = randperm(rng, nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    inds_start = ntuple(Returns(:), ndims(data) - 1)
    ## train data
    idxs_train = idxs[1:ntrain]
    train_data = data[inds_start..., idxs_train]
    train_labels = ndims(labels) == 2 ? labels[:, idxs_train] : labels[idxs_train]
    ## validation data
    idxs_val = idxs[(ntrain + 1):end]
    val_data = data[inds_start..., idxs_val]
    val_labels = ndims(labels) == 2 ? labels[:, idxs_val] : labels[idxs_val]
    (train_data, train_labels), (val_data, val_labels)
end

function train!(loss, model, train_data, opt_state, val_data; num_epochs=10)
    history = Dict("mean_batch_loss" => Float64[])
    for epoch in 1:num_epochs
        print(stderr, "")
        progress = Progress(length(train_data); desc="epoch $epoch/$num_epochs")
        total_loss = 0.0    
        for (i, Xy) in enumerate(train_data)
            batch_loss, grads = Flux.withgradient(model) do m
                loss(m(Xy[1]), Xy[2])
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += batch_loss
            ProgressMeter.next!(
                progress; showvalues = 
                [(:mean_loss, total_loss / i), (:batch_loss, batch_loss)]
            )
        end
        mean_batch_loss = total_loss / length(train_data)
        push!(history["mean_batch_loss"], mean_batch_loss)
        update_history!(history, model, train_data, "train_", loss, accuracy)
        update_history!(history, model, val_data, "val_", loss, accuracy)
    end
    println("")
    history
end

function update_history!(history::Dict, model, data, prefix::String, funcs...)
    metrics = batched_metrics(model, data, funcs...)
    for func in keys(metrics)
        metric_name = prefix * String(func)
        if !(haskey(history, metric_name))
            history[metric_name] = [metrics[func]]
        else
            push!(history[metric_name], metrics[func])
        end
        @printf "%s=%.4f; " metric_name metrics[func]
    end
    println("")
end
