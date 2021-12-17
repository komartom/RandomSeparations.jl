# RandomSeparations.jl

A similarity search method designed to help threat analysts with identification of unknown variants of known malware in unlabeled data.

## Installation
```julia
] add https://github.com/komartom/RandomSeparations.jl.git
```

## Simple example
```julia
X_pos = ones(Float32, 10, 3)     # known positive samples
X_neg = rand(Float32, 1000, 3)   # negative samples
X_xxx = [0.99f0 0.99f0 0.99f0]   # unknown positive sample
X_unl = vcat(X_neg, X_xxx)       # unlabeled samples 

X = vcat(X_unl, X_pos)
Y = vcat(falses(size(X_unl, 1)), trues(size(X_pos, 1)))

scores = RandomSeparations.run(X, Y)

findmax(scores)[2] == 1001 # index of the unknown positive sample
```