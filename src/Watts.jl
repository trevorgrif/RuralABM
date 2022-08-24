## Watt's threshold model

function threshold(x,δ)
    x .≥ δ ? 1.0 : 0.0
end

function Watts_step(state,Graph,δ)
    state = threshold.(state + threshold.(Graph*state./(Graph*ones(size(Graph,1))), δ),1)
end

function Watts(IC,Graph;tmax=30,δ=0.04)
    Sol = DataFrame(time = 1, state = [IC])
    for t in 1:(tmax-1)
        push!(Sol,[t+1,Watts_step(Sol.state[t],Graph,δ)])
    end
    return Sol
end

function time_to_infection(Watts_DF)
    fill(maximum(sum(Watts_DF.state)),size(Watts_DF.state[1],1)).-sum(Watts_DF.state)
end
