#============================================================
-------------------- Running the Model  ---------------------
============================================================#

function Run_Model!(model; duration = 0)
    # Set epidemiological data
    symptomatic(x) = x.status == :I
    recovered(x) = x.status == :R
    pop_size(x) = x.id != 0
    adata = [(symptomatic, count), (recovered, count), (pop_size, count)]

    # Run the model and extract model data
    if duration == 0
        data, mdata = run!(model, dummystep, model_step_parallel!, Is_Epidemic_Active; adata = adata, mdata = [:day])
    else
        data, mdata = run!(model, dummystep, model_step_parallel!, 12*duration; adata = adata, mdata = [:day])
    end

    TransmissionNetwork = model.TransmissionNetwork
    SCM = get_adjacency_matrix_upper(model)
    SummaryStatistics = get_epidemic_data(model, data)

    return model, data, TransmissionNetwork, SCM, SummaryStatistics
end

function Ensemble_Run_Model!(models; duration = 0)
    # Set epidemiological
    symptomatic(x) = x.status == :I
    recovered(x) = x.status == :R
    pop_size(x) = x.id != 0
    adata = [(symptomatic, count), (recovered, count), (pop_size, count)]

    # Run the model and extract model data
    if duration == 0
        data, mdata = ensemblerun!(models, agent_step!, model_step!, Is_Epidemic_Active; adata= adata, mdata = [:day])
    else
        data, mdata = ensemblerun!(models, agent_step!, model_step!, 12*duration; adata= adata, mdata = [:day])
    end

    return data, mdata
end

"""
    Construct_Town("PATH/TO/population_data.csv", "PATH/TO/business_data.csv")

Creates and Agents.jl model with space=GraphSpace and agents defined as the union of three agent types: retiree, adult, and child. Using the population_data.csv and business_data.csv, agents are assigned to workplaces and schools.

The model generated is returned alongside three dataframes describing the town structure:
     townDataSummaryDF: counts on building types and agent types
     businessStructureDF: employee data aggregated by business
     houseStructureDF: agent household assignments
"""
function Construct_Town(population_data_fp, business_data_fp)
    build_town(population_data_fp, business_data_fp)
end

"""
    Is_Epidemic_Active(model, s)

Check for any infected agents remaining in model. Returns false once an infected agent is found.
"""
function Is_Epidemic_Active(model, s)
    for agent in allagents(model)
        agent.status in [:I] && return false
    end
    return true
end

function Switch_Off_Community_Gatherings!(model)
    turn_comm_gaths_off!(model)
end

function Seed_Contagion!(model; seed_num = 1)
    infect_someone!(model,seed_num)
    model
end

function Save_Model(filepath, model)
    AgentsIO.save_checkpoint(filepath, model)
end

function Load_Model(filepath)
    AgentsIO.load_checkpoint(filepath;warn = false)
end

function Update_Agents_Attribute!(model, id_arr, attr::Symbol, new_value)
    update_agents_attr!(model, id_arr, attr, new_value)
end

function Get_Portion_Random(model, portion, CONDITIONS = [(x) -> true], DISTRIBUTION = [1.0])
    get_portion_rand(model, portion, CONDITIONS, DISTRIBUTION)
end

"""
    Get_Portion_Watts(model, target_portion)

Runs the Watts Threshold model on the adjaceny matrix of
agents in "model" until the affected portion falls within an error range of the target_portion
"""
function Get_Portion_Watts(model, portion; δ = 0.014, seed_num = 1, error_radius = 0.01, delta_shift = 0.1, MAX_NEUTRAL_EFFECT = 1000)
     get_portion_Watts(model, portion; δ, seed_num, error_radius, delta_shift, MAX_NEUTRAL_EFFECT)
end

#============================================================
------------------ Analyze Model Results  -------------------
============================================================#

function Compute_Spectral_Radius_From_Filename(filename, norm_days)
    compute_spectral_radius_from_filename(filename, norm_days)
end

function Compute_Spectral_Radius(contact_matrix)
    compute_spectral_radius(contact_matrix)
end

function Decompact_Adjacency_Matrix(filename::String)
    decompact_adjacency_matrix(filename)
end

function Get_Compact_Adjacency_Matrix(model)
    get_adjacency_matrix_upper(model)
end

function Get_Daily_Agentdata(AgentData)
    parse_to_daily!(AgentData)
end

function Get_Epidemic_Data(model, AgentData)
    get_epidemic_data(model, AgentData)
end

function Adjacency_Matrix(model)
    get_adjacency_matrix(model)
end

function Household_Adjacency_Matrix(model)
    Household_Adjacency(model)
end

function Get_Transmission_Network(model)
    model.TransmissionNetwork
end

#============================================================
--------------------- Plotting Methods  ---------------------
============================================================#

function Plot_Epidemic_Invariants(EpidemicInvariantsFilePath::String)
    plot_epidemic_invariants(EpidemicInvariantsFilePath)
end

function Save_Epidemic_Invariants_Plot(EpidemicInvariantsFilePath::String, savepath::String)
    save_epidemic_invariants_plot(EpidemicInvariantsFilePath, savepath)
end

function Serialize_Model(model)
    base64encode(serialize, model)
end

function Deserialize_Model(model)
    deserialize(IOBuffer(base64decode(model)))
end
