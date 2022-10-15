## Vaccine Module (more or less)

#============================================================
---------------------- Main Functions -----------------------
============================================================#

#============================================================
Vaccinates all agents in id_arr and returns the model.
============================================================#

function update_agents_attr!(model, id_arr, attr::Symbol, new_value)
    for id in id_arr
        setfield!(getindex(model,id), attr, new_value)
    end
    return model
end

function turn_comm_gaths_off!(model)
    model.behavior_parameters = Behavior_parameters(
        Adult_Community_Gathering = [0.0, 0.0, 0.0, 0.0, 1.0],
        Child_Community_Gathering = [0.0, 0.0, 0.0, 0.0, 1.0],
        Retiree_Community_Gathering = [0.0, 0.0, 0.0, 0.0, 1.0])
end

#============================================================
------------------ Distribution Functions -------------------
============================================================#

#============================================================
Selects a portion of the population via a random
distirbution. Returns an array of IDs.
============================================================#

function get_portion_rand(model, portion, CONDITIONS = [(x) -> true], DISTRIBUTION = [1.0])
    # Check that DISTRIBUTION length matched CONDITIONS length
    if length(CONDITIONS) != length(DISTRIBUTION)
        @warn "List length mismatch" CONDITIONS
        return []
    end

    # Collect agents satisfying COND (function)
    selectedAgentGroups::Vector{Vector{Int64}} = []
    for CONDITION in CONDITIONS
        selectedAgents::Vector{Int64} = []
        for agent in allagents(model)
            if CONDITION(agent)
                push!(selectedAgents, agent.id)
            end
        end
        push!(selectedAgentGroups, selectedAgents)
    end

    # Sample subset of agents according to DISTRIBUTION (array of proportions)
    allSelectedAgents::Vector{Int64} = []
    weightedPortion = portion*DISTRIBUTION
    numberAgents = nagents(model)
    sampleSizes = floor.(Int64, weightedPortion*numberAgents)

    for groupsIdx in eachindex(selectedAgentGroups)
        if sampleSizes[groupsIdx] > length(selectedAgentGroups[groupsIdx])
            @warn "Not enough agents in subset. Altering distribution."
            difference = sampleSizes[groupsIdx] - length(selectedAgentGroups[groupsIdx])
            sampleSizes[groupsIdx] -= difference
            try
                sampleSizes[groupsIdx+1] += difference
            catch
                @warn "Redistribution failed. Not enough agents matching CONDITIONS"
            end
        end
        append!(allSelectedAgents, sample(selectedAgentGroups[groupsIdx], sampleSizes[groupsIdx], replace = false))
    end

    return allSelectedAgents
end

#============================================================

============================================================#
"""
    get_portion_Watts(model, target_portion)

Runs the Watts Threshold model on the adjaceny matrix of
agents in "model" until the affected portion falls within an error range of the target_portion
"""
function get_portion_Watts(model, target_portion; δ = 0.014, seed_num = 1, error_radius = 0.01, delta_shift = 0.1, MAX_NEUTRAL_EFFECT = 1000)
    # Validate target_portion
    target_portion == 0.0 && return []
    target_portion > 1.0 && return []

    # Get the contact matrix of the model
    Social_Contact_Matrix = Adjacency_Matrix(model)

    # Seed an individual with the vaccine, due to random seeding, results vary remarkably...
    IC = zeros(size(Social_Contact_Matrix, 1))
    for i in sample(1:size(Social_Contact_Matrix, 1), seed_num, replace = false)
        IC[i] = 1.0
    end

    # Run the Watts Threshold model until target_portion of population is vaccinated
    Sol = DataFrame(time = 1, state = [IC])
    IsPortionMet = false
    time_step = 1
    curr_portion = 0.0
    fail_safe_count = 0

    while(true)
        # Step and test
        push!(Sol,[time_step + 1, Watts_step(Sol.state[time_step], Social_Contact_Matrix, δ)])
        time_step = time_step + 1

        # Calculate current portion
        prior_portion = curr_portion
        curr_portion = count(!iszero, Sol.state[time_step]) / size(Sol.state[time_step])[1]

        # Case 1: within target range
        if target_portion - error_radius ≤ curr_portion ≤ target_portion + error_radius
            break
        end

        # Case 2: exceeded target range --> undo last step and re-run with higher threshold
        if curr_portion > target_portion + error_radius
            δ = δ * (1.0 + delta_shift)
            deleteat!(Sol, size(Sol)[1])
            time_step = time_step - 1
            continue
        end

        # Case 3: Spread has neutralized --> lower threshold for next step
        if curr_portion == prior_portion
            δ = δ * (1.0 - delta_shift)
            fail_safe_count = fail_safe_count + 1

            if fail_safe_count > MAX_NEUTRAL_EFFECT
                @warn "Watts Distribution Failed"
                return []
            end

            continue
        end
    end

    # return affected agents
    return findall(!iszero, Sol.state[time_step])
end

function threshold(x, δ)
    x .≥ δ ? 1.0 : 0.0
end

function Watts_step(state, Graph, δ)
    state = threshold.(state + threshold.(Graph*state./(Graph*ones(size(Graph,1))), δ),1)
end

function Watts(IC, Graph; tmax=30, δ=0.04)
    Sol = DataFrame(time = 1, state = [IC])
    for t in 1:(tmax-1)
        push!(Sol,[t+1, Watts_step(Sol.state[t], Graph, δ)])
    end
    return Sol
end

function time_to_infection(Watts_DF)
    fill(maximum(sum(Watts_DF.state)),size(Watts_DF.state[1], 1)).-sum(Watts_DF.state)
end
