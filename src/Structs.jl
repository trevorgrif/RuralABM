#WARNING: All structs are initiated during town construction. Therefore, any changes will require rebuilding the town model. (Should probably make parameters changeable...)
#============================================================
------------------------- Globals --------------------------
============================================================#

VERBOSE = false

#============================================================
--------------------- Agent Structures ----------------------
============================================================#

mutable struct Adult <: AbstractAgent
    id::Int64
    pos::Int64
    age::Int64
    sex::Symbol
    home::Int64
    work::Int64
    community_gathering::Int64
    income::Int64
    shift::Tuple{Int64,Int64}
    status::Symbol
    time_infected::Rational{Int64}
    β::Float64
    contact_list::Vector{Float64}
    masked::Bool
    will_mask::Vector{Bool} #{global,local,social}
    vaccinated::Bool
    global_mask_threshold::Float64
    local_mask_threshold::Float64
    next_action::Int64
end

mutable struct Child <: AbstractAgent
    id::Int64
    pos::Int64
    age::Int64
    sex::Symbol
    home::Int64
    community_gathering::Int64
    school::Int64
    status::Symbol
    time_infected::Rational{Int64}
    β::Float64
    contact_list::Vector{Float64}
    masked::Bool
    will_mask::Vector{Bool} #{global,local,social}
    vaccinated::Bool
    global_threshold::Float64
    local_threshold::Float64
    next_action::Int64
end

mutable struct Retiree <: AbstractAgent
    id::Int64
    pos::Int64
    age::Int64
    sex::Symbol
    home::Int64
    community_gathering::Int64
    income::Int64
    status::Symbol
    time_infected::Rational{Int64}
    β::Float64
    contact_list::Vector{Float64}
    masked::Bool
    will_mask::Vector{Bool} #{gloabl,local,social}
    vaccinated::Bool
    global_threshold::Float64
    local_threshold::Float64
    next_action::Int64
end

mutable struct agent_extraction_data
    id::Int64
    will_mask::Vector{Bool}
    masked::Bool
    status::Symbol
    work::Int64
    community_gathering::Int64
end
#============================================================
--------------------- Model Paramters -----------------------
============================================================#

# Distribution structure:[Local, Global, Friends, Shopping, Nothing]
@with_kw struct Build_Behavior_Parameters
    # Setting parameters
    Adult_House::Vector{Float64} = [0.6, 0.1, 0.15, 0.1, 0.05]
    Adult_Work::Vector{Float64} = [0.7, 0.1, 0.0, 0.2, 0.0]
    Adult_Community_Gathering::Vector{Float64} = [0.8, 0.0, 0.0, 0.0, 0.2]

    Child_House::Vector{Float64} = [0.6, 0.1, 0.15, 0.1, 0.05]
    Child_School::Vector{Float64} = [0.6, 0.1, 0.2, 0.0, 0.1]
    Child_Community_Gathering::Vector{Float64} = [0.8, 0.0, 0.0, 0.0, 0.2]

    Retiree_day::Vector{Float64} = [0.5, 0.05, 0.15, 0.1, 0.2]
    Retiree_Community_Gathering::Vector{Float64} = [0.8, 0.0, 0.0, 0.0, 0.2]

    # Building distributions
    Adult_House_Distribution::Multinomial{Float64, Vector{Float64}} = Multinomial(1,Adult_House)
    Adult_Work_Distribution::Multinomial{Float64, Vector{Float64}} = Multinomial(1,Adult_Work)
    Adult_CommGath_Distribution::Multinomial{Float64, Vector{Float64}} = Multinomial(1,Adult_Community_Gathering)

    Child_House_Distribution::Multinomial{Float64, Vector{Float64}} = Multinomial(1, Child_House)
    Child_School_Distribution::Multinomial{Float64, Vector{Float64}} = Multinomial(1, Child_School)
    Child_CommGath_Distribution::Multinomial{Float64, Vector{Float64}} = Multinomial(1, Child_Community_Gathering)

    Retiree_day_Distribution::Multinomial{Float64, Vector{Float64}} = Multinomial(1, Retiree_day)
    Retiree_CommGath_Distribution::Multinomial{Float64, Vector{Float64}} = Multinomial(1, Retiree_Community_Gathering)
end

@with_kw struct Build_Disease_Parameters
    βrange::Tuple{Float64,Float64} = (1.0,2.0)
    rp::Float64 = 0.0 #re-infection probability
    vrp::Float64 = 0.15 #vaccine re-infection probability
    Infectious_period::Int64 = 20

    # Create gamma distribution pdf that Infectivity follows with time, peak infectivity at day 14 (~0.14 infections probability)
    γ_parameters::Vector{Float64} = [97.18750, 3.71875,25.625]
    incubation_period::Float64 = 6.1
    γ::Function = t -> (Gamma(γ_parameters[1], 1/γ_parameters[2]) |> (x->pdf(x,t+γ_parameters[3]-incubation_period)))
end

@with_kw struct Build_Risk_Parameters
    risk_global::Float64 = 0.0
    risk_local::Float64 = 0.0
end

@with_kw struct Build_Age_Parameters
    vax_age_thresholds::Vector{Float64} = [5,17]
    mask_age_thresholds::Vector{Float64} = [2]

    # Age ranges [5-17, 17+]
    vax_distribution_proportions::Vector{Float64} = [0.34, 0.66]

    FRIEND_RADII_DICT = Dict{DataType, Int}(
        Adult => 10,
        Child => 5,
        Retiree => 20
    )
end

@with_kw struct SIC_codes
    data = """
    SIC,Public,Client,Closed system
    1,0,0,1
    2,0,0,1
    3,0,0,0
    4,0,0,0
    5,0,0,0
    6,0,0,0
    7,0,1,0
    8,0,1,1
    9,0,1,1
    10,0,0,1
    11,0,0,0
    12,0,0,1
    13,0,0,1
    14,0,0,1
    15,0,1,1
    16,0,1,0
    17,0,1,1
    18,0,0,0
    19,0,0,0
    20,0,0,1
    21,0,0,1
    22,0,0,1
    23,0,0,1
    24,0,0,1
    25,0,0,1
    26,0,0,1
    27,0,0,1
    28,0,0,1
    29,0,0,1
    30,0,0,1
    31,0,0,1
    32,0,0,1
    33,0,0,1
    34,0,0,1
    35,0,0,1
    36,0,0,1
    37,0,0,1
    38,0,0,1
    39,0,0,1
    40,1,0,0
    41,1,0,0
    42,0,0,1
    43,1,0,0
    44,1,0,0
    45,1,0,0
    46,0,0,1
    47,1,1,0
    48,0,0,1
    49,0,0,1
    50,0,1,1
    51,0,1,1
    52,1,0,0
    53,1,0,0
    54,1,0,0
    55,1,0,0
    56,1,0,0
    57,1,0,0
    58,1,0,0
    59,1,0,0
    60,1,1,0
    61,0,1,0
    62,0,1,0
    63,0,1,0
    64,0,1,0
    65,0,1,0
    66,0,1,0
    67,0,1,0
    68,0,0,0
    69,0,0,0
    70,1,0,0
    71,0,0,0
    72,0,1,0
    73,0,1,0
    74,0,0,0
    75,0,1,0
    76,0,1,0
    77,0,0,0
    78,1,0,0
    79,1,0,0
    80,0,1,0
    81,0,1,0
    82,1,1,0
    83,0,1,0
    84,1,0,0
    85,0,0,0
    86,0,1,0
    87,0,1,0
    88,0,0,1
    89,0,1,0
    90,0,0,0
    91,0,1,1
    92,1,1,0
    93,0,1,0
    94,0,1,0
    95,0,1,0
    96,0,1,0
    97,0,1,0
    98,0,0,0
    99,0,1,0
    """
    SIC_Sheet = CSV.File(IOBuffer(data)) |> DataFrame
end
