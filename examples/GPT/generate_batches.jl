using Random
using CUDA
import Base: iterate, length, IteratorEltype

function get_shifted_batch(rng::AbstractRNG, data::AbstractVector, context_size::Int, batch_size::Int)
    indices = rand(rng, 1:(length(data)-context_size), batch_size)
    X = similar(data, context_size, batch_size)
    Y = similar(data, context_size, batch_size)
    for (j, idx) in enumerate(indices)
        X[:, j] = data[idx:(idx + context_size - 1)]
        Y[:, j] = data[(idx + 1):(idx + context_size)]
    end
    X, Y
end

get_shifted_batch(data::AbstractVector, context_size::Int, batch_size::Int) = 
    get_shifted_batch(Random.default_rng(), data, context_size, batch_size)

"""
    BatchGenerator(data; 
        batch_size=1, 
        context_size=1, 
        num_steps=length(data) ÷ context_size,
        rng=Random.default_rng()
    )

An object that generates random pairs from data, where the second object is shifted down by one row from the first.

Each output is a matrix of size `context_size × batch_size`.

For many samples ``n=num_steps``, the number of occurances of a given index will approach a binomial distribution 
with mean ``np=nc/d`` and standard deviation ``sqrt(np(1-p))=sqrt(nc/d(1-c/d))``
where ``d=length(data)`` and ``c=context_size``.
The data on the ends are slightly less likely to occur.

Based on Flux.DataLoader.

# Examples

```jldoctest
julia> data = 1:40;

julia> batch_generator = BatchGenerator(data; context_size=4, batch_size=3, num_steps=10);

julia> x, y = first(batch_generator)
([34 29 24; 35 30 25; 36 31 26; 37 32 27], [35 30 25; 36 31 26; 37 32 27; 38 33 28])

julia> for (x, y) in batch_generator
    println(size(x), " ", size(y))
    end
(4, 3) (4, 3)
(4, 3) (4, 3)
(4, 3) (4, 3)
(4, 1) (4, 1)
```
"""
struct BatchGenerator{T<:AbstractVector, R<:AbstractRNG}
    data::T
    context_size::Int
    batch_size::Int
    num_steps::Int
    rng::R
end

function BatchGenerator(
    data::AbstractVector;
    batch_size::Int = 1,
    context_size::Int=1,
    num_steps::Int=length(data) ÷ context_size,
    rng::AbstractRNG = Random.default_rng()
    )
    BatchGenerator(data, context_size, batch_size, num_steps, rng)
end

function Base.iterate(e::BatchGenerator)
    batch = get_shifted_batch(e.rng, e.data, e.context_size, e.batch_size)
    batch, e.batch_size
end

function Base.iterate(e::BatchGenerator, state)
    if state >= e.num_steps
        return nothing
    end
    batch_size = state + e.batch_size > e.num_steps ? e.num_steps - state : e.batch_size
    batch = get_shifted_batch(e.rng, e.data, e.context_size, batch_size)
    batch, state + batch_size
end

Base.IteratorEltype(::BatchGenerator) = Base.EltypeUnknown()

Base.length(e::BatchGenerator) = ceil(Int, e.num_steps / e.batch_size)

count_observations(data::D) where {D<:BatchGenerator} = count_observations(data.data)