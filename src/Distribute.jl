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

    for groupsIdx in 1:length(selectedAgentGroups)
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
Runs the Watts Threshold model on the adjaceny matrix of
agents in "model" until the affected portion meets or
exceeds "portion".

WARNING: May fail if Watts threshold stabalized before
reaching the desired poriton. Need to make a smarter function
to prevent this...

Plan to fix: adjust get_portion_watts() to take in seeding
list and make vaccinate_Watts!() randomly choose seed_num
agents and check against get_portion_Watts. vaccinate_Watts()
will try a default number of times to get a valid seeding.
============================================================#
###SHOULD BE IN WATTS.JL eh... maybe watts should be in here...###
function get_portion_Watts(model, portion; δ = 0.014, seed_num = 1, VERBOSE = false)
    # Get the contact matrix of the model
    Contact_Matrix_Person = Get_Adjacency_Matrix(model)

    # Seed an individual with the vaccine, due to random seeding, results vary remarkably...
    IC = zeros(size(Contact_Matrix_Person,1))
    for i in sample(1:size(Contact_Matrix_Person,1), seed_num, replace = false)
        IC[i] = 1.0
    end

    # Run the Watts Threshold model until desired portion of population is vaccinated
    Sol = DataFrame(time = 1, state = [IC])
    IsPortionMet = false
    time_step = 1
    while(!IsPortionMet)
        # Printing current portion
        curr_portion = count(!iszero, Sol.state[time_step]) / size(Sol.state[time_step])[1]
        if VERBOSE
            print(curr_portion)
            print("\n")
        end

        # Check if desired portion is vaccinated, otherwise step again
        if curr_portion >= portion
            IsPortionMet = true
            break
        else
            push!(Sol,[time_step + 1, Watts_step(Sol.state[time_step], Contact_Matrix_Person, δ)])
        end
        time_step = time_step + 1
    end

    # return affected agents
    return findall(!iszero, Sol.state[time_step])
end

#============================================================
--------------------- Helper Functions ----------------------
============================================================#

#============================================================
Computes the minimum seed number (see vaccinate_Watts! params)
such that the entire population gets vaccinated.
============================================================#

function compute_min_seed_num(model, δ)
    seed = 1
    graph = Get_Adjacency_Matrix(model)

    while(true)
        result = check_portion_Watts(graph, δ, seed)
        if result == 1.0
            return seed
        else
            seed = seed + 1
        end

    end
    return seed
end


#============================================================
A helper function to compute_min_seeding. Runs the Watts
threshold model on the graph and returns the portion of the
population who recieved the vaccine.
============================================================#
function check_portion_Watts(graph, δ, seed_num)
    # Seed the vaccine
    IC = zeros(size(graph,1))
    for i in 1:seed_num
        IC[i] = 1.0
    end

    # Run the Watts Threshold model until pass or fail
    Sol = DataFrame(time = 1, state = [IC])
    time_step = 1

    # While loop will be escaped since the Watts threshold is monotone increasing
    while(true)

        # Store the prior portion vaccinated, run the next Watts step, and then save
        old_portion = count(!iszero, Sol.state[time_step]) / size(Sol.state[time_step])[1]
        push!(Sol,[time_step + 1, Watts_step(Sol.state[time_step], graph, δ)])
        time_step = time_step + 1
        curr_portion = count(!iszero, Sol.state[time_step]) / size(Sol.state[time_step])[1]

        # Check if model has stabilized
        if curr_portion == old_portion
            return curr_portion
        elseif curr_portion == 1.0
            return 1.0
        end
    end
end
