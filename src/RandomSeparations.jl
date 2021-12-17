module RandomSeparations

using Statistics, Random, StatsBase

export run


struct Options
    n_subfeat::Int
    n_thresholds::Int
    zero_score::Float64
    pos_subsample::Float64
    ignore_samples::Set{Int}
end


struct Split
    n_separated::Int
    is_left::Bool
    feature::Int
    threshold::Float32
end

Split() = Split(0, true, 0, 0.0f0)



function make_split(V, labels, n_pos_samples, n_unl_samples, feature, threshold)
    n_left_pos, n_left_unl = 0, 0
    for (vv, ll) in zip(V, labels)
        if vv < threshold
            if ll
                n_left_pos += 1
            else
                n_left_unl += 1
            end
        end
    end
    n_right_pos = n_pos_samples - n_left_pos
    n_right_unl = n_unl_samples - n_left_unl

    if n_left_pos == n_pos_samples
        return Split(n_right_unl, true, feature, threshold)
    end

    if n_right_pos == n_pos_samples
        return Split(n_left_unl, false, feature, threshold)
    end

    return Split()
end



function score(X, Y, opt, seed)
    rng = MersenneTwister(seed)

    scores = Int[]
    iterations = zeros(Int, length(Y))

    features = collect(1:size(X, 2))

    pos_samples = shuffle!(rng, [ss for ss in 1:length(Y) if (!(ss in opt.ignore_samples) && Y[ss])])
    pos_subsample_set = Set(pos_samples[1:round(Int, length(pos_samples) * opt.pos_subsample)])

    samples = [ss for ss in 1:length(Y) if (!(ss in opt.ignore_samples) && (Y[ss] ? (ss in pos_subsample_set) : true))]
    
    # iterations
    while true
        
        labels = Y[samples]
        n_pos_samples = sum(labels)
        n_unl_samples = length(labels) - n_pos_samples

        if n_pos_samples == 0 || n_unl_samples == 0
            break
        end

        best_split = Split()
        first_usable_feat = 1
        last_nonconst_feat = length(features)
        V = Vector{Float32}(undef, length(samples))

        mtry = 1
        while (mtry <= opt.n_subfeat) && (first_usable_feat <= last_nonconst_feat)

            ff = rand(rng, first_usable_feat:last_nonconst_feat)
            feature = features[ff]
            features[ff], features[first_usable_feat] = features[first_usable_feat], features[ff]
            first_usable_feat += 1

            minv, maxv = Inf32, -Inf32
            for (ii, ss) in enumerate(samples)
                V[ii] = X[ss, feature]
                if minv > V[ii]
                    minv = V[ii]
                end
                if maxv < V[ii]
                    maxv = V[ii]
                end
            end

            if minv == maxv
                first_usable_feat -= 1
                features[first_usable_feat], features[last_nonconst_feat] = features[last_nonconst_feat], features[first_usable_feat]
                last_nonconst_feat -= 1
                continue
            end

            for _ in 1:opt.n_thresholds
                threshold = minv + (maxv - minv) * rand(rng, Float32)
                split = make_split(V, labels, n_pos_samples, n_unl_samples, feature, threshold)
                if split.n_separated > best_split.n_separated
                    best_split = split
                end
            end

            mtry += 1
        end

        push!(scores, best_split.n_separated)
        interation = length(scores)

        if best_split.n_separated == 0
            for (ss, ll) in zip(samples, labels)
                if !ll
                    iterations[ss] = interation
                end
            end
            break
        end

        nn = 1
        new_samples = Vector{Int}(undef, length(samples) - best_split.n_separated)
        for ss in samples
            if best_split.is_left == (X[ss, best_split.feature] < best_split.threshold)
                new_samples[nn] = ss 
                nn += 1
            else
                iterations[ss] = interation
            end
        end

        samples = new_samples
        features = features[1:last_nonconst_feat]
    end

    return [ii > 0 ? (scores[ii] > 0 ? 1.0 / scores[ii] : opt.zero_score) : 0 for ii in iterations]

end



function run(
            X               ::Matrix{Float32}, 
            Y               ::AbstractArray{Bool}; 
            n_repetitions   ::Int=100, 
            n_subfeat       ::Int=0, 
            n_thresholds    ::Int=8, 
            zero_score      ::Float64=1.0, 
            pos_subsample   ::Float64=1.0, #0.67
            ignore_samples  ::Set{Int}=Set{Int}(), 
            seed            ::Int=1234)

    @assert any(Y)
    @assert !all(Y)
    @assert size(X, 1) == length(Y)
    @assert zero_score >= 1.0

    if !(1 <= n_subfeat <= size(X, 2)) 
        n_subfeat = round(Int, sqrt(size(X, 2)))
    end

    opt = Options(n_subfeat, n_thresholds, zero_score, pos_subsample, ignore_samples)

    repetitions = Vector{Vector{Float64}}(undef, n_repetitions)
    seeds = abs.(rand(MersenneTwister(seed), Int, n_repetitions))
    Threads.@threads for rr in 1:n_repetitions
        repetitions[rr] = score(X, Y, opt, seeds[rr])
    end

    # return reduce((x, y) -> min.(x, y), repetitions)
    return reduce((x, y) -> (x .+= y), repetitions) / n_repetitions

end

end #module