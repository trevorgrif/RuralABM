#============================================================
------------------ MAIN STEPPING FUNCTIONS ------------------
agent_step! gets called at each time step in the model (every
hour). The method is overloaded for each type of agent in
the model.

After all agents in agents.scheduler have been activated,
model_step! is called.
============================================================#

#============================================================
On weekends, adults take any of the following actions:
    Community_Gathering
    Shopping
    Nothing
On weekdays, adults either go to their workplace or stay home
and take any of the following actions:
    Socialize Global
    Socialize Local
    Hang with Friends
    Go Shopping
    Nothing
Lastly, infection parameters and statistics are updated
============================================================#
function agent_step_decide!(adult::Adult, model)
    # Weekday
    if mod(model.day,6) != 0
        # If at home then spin home paramters, otherwise spin work parameters
        if get_prop(model.space.graph,adult.pos,:Type) == :House
            adult.next_action = spin(model.behavior_parameters.Adult_House_Distribution)
        else
            adult.next_action = spin(model.behavior_parameters.Adult_Work_Distribution)
        end
        return
    end

    # Weekend Community Gathering
    if (1 ≤ model.time ≤ 4) && (adult.community_gathering != 0)
        #Spin community_gathering parameters for family
        adult.next_action = spin(model.behavior_parameters.Adult_CommGath_Distribution)
        return
    end

    # Weekend No Community Gathering
    adult.next_action = spin(model.behavior_parameters.Adult_House_Distribution)
    return
end

#============================================================
On weekends, retirees take any of the following actions:
    Community_Gathering
    Shopping
    Nothing
On weekdays, retirees take any of the following actions:
    Socialize Global
    Socialize Local
    Hang with Friends
    Go Shopping
    Nothing
Lastly, infection parameters and statistics are updated
============================================================#
function agent_step_decide!(retiree::Retiree, model)
    # Weekday
    if mod(model.day,6) != 0
        retiree.next_action = spin(model.behavior_parameters.Retiree_day_Distribution)
        return
    end

    # Weekend Community Gathering
    if (1 ≤ model.time ≤ 4) && (retiree.community_gathering != 0)
        # Spin community_gathering parameters
        retiree.next_action = spin(model.behavior_parameters.Retiree_CommGath_Distribution)
        return
    end

    # Weekend No Community Gathering
    retiree.next_action = spin(model.behavior_parameters.Retiree_day_Distribution)
    return
end

#============================================================
On weekends,attend their household community gatherings. On weekdays,
children attend school and take any of the following actions:
    Socialize Global
    Socialize Local
    Hang with Friends
    Go Shopping
    Nothing
Lastly, infection parameters and statistics are updated
============================================================#
function agent_step_decide!(child::Child, model)
    # Weekday
    if mod(model.day,6) != 0
        # Move child to school if school hours apply
        #9 ≥ model.time ≥ 3 ? move_agent!(child, child.school, model) : move_agent!(child, child.home, model)

        # If at home spin home parameters, otherwise, spin school parameters
        if get_prop(model.space.graph, child.pos,:Type) == :House
            child.next_action = spin(model.behavior_parameters.Child_House_Distribution)
        else
            child.next_action = spin(model.behavior_parameters.Child_School_Distribution)
        end
        return
    end

    # Weekend Community Gathering
    if (1 ≤ model.time ≤ 4) && (child.community_gathering != 0)
        # Move agent and children to gathering
        #move_agent!(child, child.community_gathering, model)

        #Spin community_gathering parameters for family
        child.next_action = spin(model.behavior_parameters.Child_CommGath_Distribution)
        return
    end

    # Weekend No Community Gathering
    child.next_action = spin(model.behavior_parameters.Child_House_Distribution)
    return
end

function agent_step_reset!(adult::Adult, model)
    # Weekday
    if mod(model.day,6) != 0
        # Move agents to home or work depending on assigned shift
        adult.shift[2] ≥ model.time ≥ adult.shift[1] ? move_agent!(adult, adult.work, model) : move_agent!(adult,adult.home,model)
        return
    end

    # Weekend Community Gathering
    if (1 ≤ model.time ≤ 4) && (adult.community_gathering != 0)
        # Move agent and children to gathering
        move_agent!(adult, adult.community_gathering, model)
        return
    end

    # Weekend No Community Gathering
    move_agent!(adult, adult.home, model)
    return
end

function agent_step_reset!(retiree::Retiree, model)
    # Weekday
    if mod(model.day,6) != 0
        move_agent!(retiree, retiree.home, model)
        return
    end

    # Weekend Community Gathering
    if (1 ≤ model.time ≤ 4) && (retiree.community_gathering != 0)
        # Move agent
        move_agent!(retiree, retiree.community_gathering, model)
        return
    end

    # Weekend No Community Gathering
    move_agent!(retiree, retiree.home, model)
    return
end

function agent_step_reset!(child::Child, model)
    # Weekday
    if mod(model.day,6) != 0
        # Move child to school if school hours apply
        9 ≥ model.time ≥ 3 ? move_agent!(child, child.school, model) : move_agent!(child, child.home, model)
        return
    end

    # Weekend Community Gathering
    if (1 ≤ model.time ≤ 4) && (child.community_gathering != 0)
        # Move agent and children to gathering
        move_agent!(child, child.community_gathering, model)
        return
    end

    # Weekend No Community Gathering
    move_agent!(child, child.home, model)
    return
end

function update_agent_infection!(agent, model)
    agent.time_infected += 1//12
    recover_or_die!(agent, model)
    return true
end

function agent_step_interact!(agent, model)
    agent.next_action == 0 && return
    DoSomething!(agent.next_action, agent, model)
end

#============================================================
Update hourly parameters
============================================================#
function model_step_parallel!(model)
    # Update agent attributes
    Threads.@threads for id in collect(model.scheduler(model))
        agent = model[id]
        agent.next_action = 0
        agent.status == :I && update_agent_infection!(agent, model)
    end

    # Move agent to default location for the hour
    for id in model.scheduler(model)
        agent_step_reset!(model[id], model)
    end

    # In parallel update agents positions and `next_action`, on a single thread run interactions between agents
    Threads.@threads for id in collect(model.scheduler(model))
        agent_step_decide!(model[id], model)
    end

    for id in model.scheduler(model)
        agent_step_interact!(model[id], model)
    end

    # Model Timer
    if model.time == 0
        Get_EOD_Data(model) # Gather any end of day data for extraction later
        model.day += 1 # Increase the day counter
    end
    model.time = mod(model.time+1, 12) # Increase the hour by 1 (12 hour days)
    model.model_steps += 1
end

#============================================================
------------------- Data Extraction Functions ---------------
============================================================#

function Get_EOD_Data(model)
    push!(model.Agent_Extraction_Data, [Get_Agent_Extraction_Data_Bridge(model,agentid) for agentid in 1:model.init_pop_size])
end

function Get_Agent_Extraction_Data_Bridge(model, agentid)
    # Deal with dead agents first
    if agentid in model.DeadAgents[:,1]
        return agent_extraction_data(
        agentid,
        zeros(3),
        false,
        :D,
        0,
        0
        )
    end
    # Otherwise call Get_Agent_Extraction_Data
    return Get_Agent_Extraction_Data(model, getindex(model, agentid))
end

function Get_Agent_Extraction_Data(model, agent::Child)
    return agent_extraction_data(
        agent.id,
        agent.will_mask,
        agent.masked,
        agent.status,
        0,
        0
        )
end

function Get_Agent_Extraction_Data(model, agent::Adult)
    return agent_extraction_data(
        agent.id,
        agent.will_mask,
        agent.masked,
        agent.status,
        agent.work,
        agent.community_gathering
        )
end

function Get_Agent_Extraction_Data(model, agent::Retiree)
    return agent_extraction_data(
        agent.id,
        agent.will_mask,
        agent.masked,
        agent.status,
        0,
        agent.community_gathering
        )
end

#============================================================
---------------------- Decision Functions -------------------
============================================================#
#============================================================
Returns a random integer with bias as defined in parameter
distributions (see Town.jl)
============================================================#
function spin(dist)
    findall(!iszero,rand(dist))[1]
end

#============================================================
Forking function for calling action functions (see parameter
definitions in Town.jl for probability distributions)
============================================================#
function DoSomething!(action, agent, model)
        @match action begin
        1 => socialize_local!(agent, model)
        2 => socialize_global!(agent, model)
        3 => hang_with_friends!(agent, model)
        4 => go_shopping!(agent, model)
        end
end

#============================================================
------------------------ Action Functions -------------------
============================================================#

#============================================================
Have agent interact with another agent in a nearby location
regardless of any other factors.
============================================================#
function socialize_local!(agent, model)
    # Apply masking behavior
    agent.masked = agent.will_mask[2]

    # collect agents at location and interact
    stranger_list = nearby_ids(agent, model)
    isempty(stranger_list) && return
    friend_id = rand(stranger_list)
    interact!(agent, model[friend_id], model)
end

#============================================================
All Socialize_Global (all): interact agent with random agent
of similar age
============================================================#
function socialize_global!(agent, model)
    agent.masked = agent.will_mask[1]
    # Grab a friend
    friend = random_agent(model, x-> abs(x.age.-agent.age) < model.age_parameters.FRIEND_RADII_DICT[typeof(agent)])
    isnothing(friend) && return
    friend.id == agent.id && return

    # Move agent to friends location -> interact
    move_agent!(agent, friend.pos, model)
    interact!(agent,friend,model)
end

#============================================================
Sends an agent to a business location where they interact
with other agents at the business.
============================================================#
function go_shopping!(agent,model)
    # Apply masking behavior
    agent.masked = agent.will_mask[1]

    # Select a business
    loc = rand(model.business)

    # Move agent to business and interact (garunteed for workers to be at location)
    original_location = agent.pos
    move_agent!(agent, loc, model)
    stranger_idx_list = nearby_ids(agent, model)
    isempty(stranger_idx_list) && return
    friend_id = rand(stranger_idx_list)
    interact!(agent, model[friend_id], model)

    # Move agent back to original location if business is not public facing
    if  get_prop(model.space.graph,loc,:business_type)[1] != 1
        move_agent!(agent, original_location, model)
    end
end

#============================================================
Agent interacts with someone at a 'friend's location.
============================================================#
function hang_with_friends!(agent,model)
    # Apply masking bahvior
    agent.masked = agent.will_mask[3]

    # Filter out dead agents
    friend_ids = copy(agent.contact_list)
    for dead_agent_id in model.DeadAgents.Agent
        friend_ids[dead_agent_id] = 0.0
    end

    # If no friends exist, socialize_local
    if sum(agent.contact_list) == 0
        socialize_local!(agent,model)
        return
    end

    # Selecting a random contact with a bias towards contacts with higher interaction counts
    friend_ids = Multinomial(1,friend_ids/sum(friend_ids)) |> rand

    # Return agent of selected contact
    friend = getindex(model,findfirst(x -> !iszero(x),friend_ids))

    # If friends exist, move agent to friends location
    move_agent!(agent, friend.pos, model)

    socialize_local!(agent,model)
    nothing
end

#============================================================
--------------------- Disease Functions ---------------------
============================================================#
#=
Wearing a mask during an interaction reduced the probability of infection by 1/4 (make variable)
------ Infection Status ------
:S = Susceptible (Default)
:I = Sypmtomatic Infection
:R = Recovered
:V = Vaccinated
=#
function infect_someone!(model, n::Int64)
    for i in 1:n
        # Should probably check that a :S agent exist
        agent = random_agent(model, x -> x.status == :S)
        agent.status = :I
    end
    for infected in filter(x -> x.status == :I, collect(allagents(model)))
        push!(model.TransmissionNetwork,[infected.id, 0, 0])
    end
end

function infect!(agent, contact, model)
    rand(model.rng) > model.disease_parameters.γ(contact.time_infected)*contact.β*4.0^(-contact.masked) && return
    agent.status = :I
    push!(model.TransmissionNetwork, [agent.id, contact.id, model.time+12*model.day])
end

function recover_or_die!(agent, model)
    # Recover
    if agent.time_infected ≥ model.disease_parameters.Infectious_period
        agent.status = :R
        agent.time_infected = 0
        return
    end

    # Die 
    if rand(model.rng) < get_IFR(agent.age)*model.disease_parameters.γ(agent.time_infected)/(model.disease_parameters.Infectious_period*12)
        push!(model.DeadAgents,[agent.id,agent.home,agent.contact_list])
        kill_agent!(agent, model)
        return
    end
end

function interact!(agent, contact, model)
    if (agent.pos == agent.home | contact.pos == contact.home)
        agent.contact_list[contact.id] += 1
        contact.contact_list[agent.id] += 1
    elseif !agent.masked & !contact.masked
        agent.contact_list[contact.id] += 1
        contact.contact_list[agent.id] += 1
    else
        agent.contact_list[contact.id] += 0.25
        contact.contact_list[agent.id] += 0.25
    end
    ## Infection dynamics
    count(a.status == :I for a in (agent,contact)) != 1 && return
    infected, healthy = agent.status == :I ? (agent,contact) : (contact,agent)
    healthy.status in [:R] && rand(model.rng) > model.disease_parameters.rp && return
    healthy.status in [:V] && rand(model.rng) > model.disease_parameters.vrp && return
    infect!(healthy,infected,model)
    nothing
end

#============================================================
--------------------- Helper Functions ----------------------
============================================================#
#============================================================
Returns the children of an agent
============================================================#
function get_children(agent,model)
    filter(x -> typeof(x) == Child, collect(agents_in_position(agent.pos, model)))
end

#============================================================
Grabs a random business from the set model.business
(see Town.jl for model construction)
============================================================#
function compute_global_risk!(model)
    #nRecovered = count(agents->(agents.status == :I), collect(allagents(model)))
    #model.risk_parameters.risk_global = nRecovered/nagents(model)
end

function compute_local_risk!(model)
    #nRecovered = count(agents->(agents.status == :I), collect(allagents(model)))
    #model.risk_parameters.risk_local = nRecovered/nagents(model)
end

function update_agent_behavior!(model)
    for agent in allagents(model)
        agent.global_mask_threshold < model.risk_parameters.global_risk ? agent.will_mask[1] = true : agent.will_mask[1] = false
        agent.local_mask_threshold < model.risk_parameters.local_risk ? agent.will_mask[2] = true : agent.will_mask[2] = false
    end
end

get_IFR(age) = 10^(-3.27+0.0524*age)/100
#end #RuralABM
