#============================================================
-------------------------- Modules --------------------------
============================================================#
# External Modules

# Local Modules

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
function agent_step!(adult::Adult, model)
    # Distribution of Church goers?
    # Determine activity for the hour
    if adult.status != :I
        if mod(model.day,6) == 0 # Weekend
            children = get_children(adult,model)
            if (4 ≥ model.time ≥ 1) && (adult.community_gathering != 0)
                # Move agent and children to gathering
                move_agent!(adult, adult.community_gathering, model)
                for child in children
                    move_agent!(child,adult.community_gathering,model)
                end

                #Spin community_gathering parameters for family
                DoSomething!(spin(model.behavior_parameters.Adult_CommGath_Distribution), adult, model)
                for child in children
                    DoSomething!(spin(model.behavior_parameters.Child_CommGath_Distribution), child, model)
                end
            else
                #Spin normal parameters for family
                DoSomething!(spin(model.behavior_parameters.Adult_House_Distribution), adult, model)
                for child in children
                    DoSomething!(spin(model.behavior_parameters.Child_House_Distribution), child, model)
                end
            end
        else # Weekday
            # Move agents to home or work depending on assigned shift
            adult.shift[2] ≥ model.time ≥ adult.shift[1] ? move_agent!(adult, adult.work, model) : move_agent!(adult,adult.home,model)

            # If at home then spin home paramters, otherwise spin work parameters
            if get_prop(model.space.graph,adult.pos,:Type) == :House
                DoSomething!(spin(model.behavior_parameters.Adult_House_Distribution),adult,model)
            else
                DoSomething!(spin(model.behavior_parameters.Adult_Work_Distribution),adult,model)
            end
        end
    end
    # Update infection parameters and statistics
    if adult.status ∈ [:I]
        adult.time_infected += 1//12
        recover_or_die!(adult,model)
    end
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
function agent_step!(geezer::Retiree,model)
    # Determine actions for the hour
    if geezer.status != :I
        if mod(model.day,6) == 0 # Weekend
            if (4 ≥ model.time ≥ 1) && (geezer.community_gathering != 0)
                # Move agent
                move_agent!(geezer, geezer.community_gathering, model)

                #Spin community_gathering parameters
                DoSomething!(spin(model.behavior_parameters.Retiree_CommGath_Distribution), geezer, model)
            else
                #Spin normal parameters
                DoSomething!(spin(model.behavior_parameters.Retiree_day_Distribution), geezer, model)
            end
        else # Weekday
            move_agent!(geezer, geezer.home, model)
            DoSomething!(spin(model.behavior_parameters.Retiree_day_Distribution), geezer, model)
        end
    end

    # Update infection parameters and statistics
    if geezer.status ∈ [:I]
        geezer.time_infected += 1//12
        recover_or_die!(geezer,model)
    end
end

#============================================================
On weekends, children follow their parents. On weekdays,
children attend school and take any of the following actions:
    Socialize Global
    Socialize Local
    Hang with Friends
    Go Shopping
    Nothing
Lastly, infection parameters and statistics are updated
============================================================#
function agent_step!(child::Child, model)
    #Determine activity for the hour
    if child.status != :I
        if mod(model.day,6) == 0 # Weekend
            # Follows Parent
            return
        else # Weekday
            # Move child to school if school hours apply
            9 ≥ model.time ≥ 3 ? move_agent!(child, child.school, model) : move_agent!(child, child.home, model)

            # If at home spin home parameters, otherwise, spin school parameters
            if get_prop(model.space.graph, child.pos,:Type) == :House
                DoSomething!(spin(model.behavior_parameters.Child_House_Distribution),child,model)
            else
                DoSomething!(spin(model.behavior_parameters.Child_School_Distribution),child,model)
            end
        end
    end

    # Update infection parameters and statistics
    if child.status ∈ [:I]
        child.time_infected += 1//12
        recover_or_die!(child,model)
    end
end

#============================================================
Update hourly parameters
============================================================#
function model_step!(model)
    if model.time == 0
        Get_EOD_Data(model) # Gather any end of day data for extraction later
        model.day += 1 # Increase the day counter
    end
    model.time = mod(model.time+1,12) # Increase the hour by 1 (12 hour days)
    mod(model.day,7) == 0 && compute_global_risk!(model) ## Once a week re-compute global risk
    mod(model.day,7) == 0 && compute_local_risk!(model) ## Once a week re-compute local risk
    model.model_steps += 1
    #update_agent_behavior(model)
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
function DoSomething!(action,agent,model)
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
    agent.masked = agent.will_mask[2]
    #If isolated, then do nothing
    isempty(nearby_ids(agent,model)) && return

    interact!(agent,rand(collect(agents_in_position(agent.pos, model))), model)
end

#============================================================
All Socialize_Global (all): interact agent with random agent
of similar age
============================================================#
function socialize_global!(agent::Adult,model)
    agent.masked = agent.will_mask[1]
    # Grab a friend
    friend = random_agent(model, x-> abs(x.age.-agent.age) < 10)
    (isequal(friend, agent) || isnothing(friend)) && return

    # Move agent to friends location -> interact -> return to prior position
    curr_loc = agent.pos
    move_agent!(agent, friend.pos, model)
    interact!(agent,friend,model)
    move_agent!(agent, curr_loc, model)
end

function socialize_global!(agent::Child,model)
    agent.masked = agent.will_mask[1]
    # Grab a friend
    friend = random_agent(model, x-> abs(x.age.-agent.age) < 5)
    (isequal(friend, agent) || isnothing(friend)) && return

    # Move agent to friends location -> interact -> return to prior position
    curr_loc = agent.pos
    move_agent!(agent, friend.pos, model)
    interact!(agent,friend,model)
    move_agent!(agent, curr_loc, model)
end

function socialize_global!(agent::Retiree,model)
    agent.masked = agent.will_mask[1]
    # Grab a friend
    friend = random_agent(model, x-> abs(x.age.-agent.age) < 20)
    (isequal(friend, agent) || isnothing(friend)) && return

    # Move agent to friends location -> interact -> return to prior position
    curr_loc = agent.pos
    move_agent!(agent, friend.pos, model)
    interact!(agent,friend,model)
    move_agent!(agent, curr_loc, model)
end

#============================================================
Sends an agent to a business location where they interact
with other agents at the business.
============================================================#
function go_shopping!(agent,model)
    agent.masked = agent.will_mask[1]
    # Find a valid location was chosen
    loc = find_business(agent,model)
    isnothing(loc) && return # Location is only nothing if no businesses have employees there yet (should occur rarely)

    # Move agent to gathering and interact
    if  get_prop(model.space.graph,loc,:business_type)[1] == 1
            move_agent!(agent,loc,model)
            interact!(agent,rand(collect(agents_in_position(agent.pos, model))),model)
    else
        interact!(agent,rand(collect(agents_in_position(loc, model))),model)
    end
    nothing
end

function go_shopping!(agent::Adult,model) # Can probably merge go_shopping functions with the GetChildren structure
    agent.masked = agent.will_mask[1]
    agent.pos == agent.work ? GetChildren = false : GetChildren = true

    GetChildren ? children = get_children(agent,model) : nothing

    # Find a valid location was chosen
    loc = find_business(agent,model)
    (loc == agent.pos || isnothing(loc)) && return

    # Move agent and children to gathering and interact
    if  get_prop(model.space.graph,loc,:business_type)[1] == 1
            move_agent!(agent,loc,model)
            if GetChildren
                for child in children
                    move_agent!(child,loc,model)
                    #interact!(child,agent,model)
                end
            end
            interact!(agent,rand(collect(agents_in_position(agent.pos,model))),model)
    else
        interact!(agent,rand(collect(agents_in_position(loc,model))),model)
    end

    nothing
end

#============================================================
SHOULD have agent interact with friends. Instead...

Has agent interact with random agent near a friends location.
============================================================#
function hang_with_friends!(agent,model)
    agent.masked = agent.will_mask[3]
    # If friends exist, move agent to friends location
    sum(agent.contact_list) ≠ 0 && move_agent!(agent,get_friend(agent,model).pos,model)

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

function infect!(agent,contact,model)
    rand(model.rng) > model.disease_parameters.γ(contact.time_infected)*contact.β*4.0^(-contact.masked) && return
    agent.status = :I
    push!(model.TransmissionNetwork, [agent.id, contact.id, model.time+12*model.day])
end

function recover_or_die!(agent,model)
    if agent.time_infected ≥ model.disease_parameters.Infectious_period
        agent.status = :R
        agent.time_infected = 0
        return
    end
    if rand(model.rng) < get_IFR(agent.age)*model.disease_parameters.γ(agent.time_infected)/(model.disease_parameters.Infectious_period*12)
        push!(model.DeadAgents,[agent.id,agent.home,agent.contact_list])
        kill_agent!(agent,model)
        return
    end
end

function interact!(agent,contact,model)
    if (agent.pos == agent.home | contact.pos == contact.home)
        agent.contact_list[contact.id] += 1
        contact.contact_list[agent.id] += 1
    elseif !agent.masked & !contact.masked
        agent.contact_list[contact.id] += 1
        contact.contact_list[agent.id] += 1
    else
        agent.contact_list[contact.id] += 0.25
        contact.contact_list[contact.id] += 0.25
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

### USE DEAD AGENTS LIST TO CHECK FOR DEAD FRIENDS
#============================================================
Returns a friend of the passed in agent.

Friends are defined as agents who have any contact with each
other. Friends are more likely to be selected if they have a
higher contact count with the agent.
============================================================#
function get_friend(agent,model)
    friend_ids = get_living_contacts(agent,model)

    # Selecting a random contact with a bias towards contacts with higher interaction counts
    friend_ids = Multinomial(1,friend_ids/sum(friend_ids)) |> rand

    # Return agent of selected contact
    return getindex(model,findfirst(x -> !iszero(x),friend_ids))
end

#============================================================
Filters out the dead contacts by returning a vector of length
equal to the contact_list length with the id for living
contacts and 0 for dead contacts.
============================================================#
function get_living_contacts(agent,model)
    # Set length of Id_List and instansiate with 0s
    Id_List = SparseVector(zeros(size(agent.contact_list)))

    # Filter out dead contacts
    for index in filter(x -> x ∉ model.DeadAgents.Agent,agent.contact_list.nzind)
        Id_List[index] = agent.contact_list[index]
    end

    return Id_List
end

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
function find_business(agent,model)
    try
        filter(x -> !isempty(x,model),model.business) |> rand #Not sure why we filter by empty locations
    catch
        nothing
    end
end

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
