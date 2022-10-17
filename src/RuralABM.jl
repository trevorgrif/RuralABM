module RuralABM
export
    Ensemble_Run_Model!,
    Is_Epidemic_Active,
    Compute_Spectral_Radius_From_Filename,
    Compute_Spectral_Radius,
    Decompact_Adjacency_Matrix,
    Save_Epidemic_Invariants_Plot,
    Plot_Epidemic_Invariants,
    Get_Daily_Agentdata,
    Get_Compact_Adjacency_Matrix,
    Get_Epidemic_Data,
    Get_Transmission_Network,
    Switch_Off_Community_Gatherings!,
    Seed_Contagion!,
    Serialize_Model,
    Deserialize_Model,
    Construct_Town,
    Run_Model!,
    Save_Model,
    Load_Model,
    Household_Adjacency_Matrix,
    Adjacency_Matrix,
    Update_Agents_Attribute!,
    Get_Portion_Random,
    Get_Portion_Watts

    using Distributed
    using Parameters
    using JLD, XLSX, CSV
    using Agents
    using Graphs, MetaGraphs
    using DataFrames
    using Random, Distributions
    using SparseArrays
    using Match
    using StatsBase, StatsPlots
    using Serialization, Base64
    using PlotlyJS, Printf, Plots
    using LinearAlgebra
    using Ripserer

    include("Structs.jl")
    include("Interface.jl")
    include("AgentBehavior.jl")
    include("Distribute.jl")
    include("Town.jl")
    include("Matrices.jl")
    include("Analysis.jl")
    include("PlotABM.jl")

    ## Unused and probably broken
    #include("Ideologies.jl")
    #include("Age_Structured_Contacts.jl")

end # Module
