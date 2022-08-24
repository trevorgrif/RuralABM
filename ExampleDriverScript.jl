#=
Example driver script for running the rural-abm model
=#

# Load rural-abm
include("src/RuralABM.jl")
using Pkg, .RuralABM
Pkg.activate(".")

# Initialize town with household, work, and school assignments
model, townDataSummaryDF, businessStructureDF, houseStructureDF = Construct_Town("data/example_towns/small_town/population.csv", "data/example_towns/small_town/businesses.csv")

# Run the model without any contagion to converge the social network
length_to_run_in_days = 30
Run_Model!(model; duration = length_to_run_in_days)

# Apply vaccination and masking behaviors to certain age ranges
portion_will_mask = 0.3
portion_vaxed = 0.2
mask_id_arr = Get_Portion_Random(model, portion_will_mask, [(x)->x.age >= 2])
vaccinated_id_arr = Get_Portion_Random(model, portion_vaxed, [(x)-> x.age > 4 && x.age < 18, (x)->x.age >= 18], [0.34, 0.66])

Update_Agents_Attribute!(model, mask_id_arr, :will_mask, [true, true, true])
Update_Agents_Attribute!(model, vaccinated_id_arr, :status, :V)
Update_Agents_Attribute!(model, vaccinated_id_arr, :vaccinated, true)

# Run the model with contagion until the count of infected agents is zero
Seed_Contagion!(model) # set seed_num = x for x seedings. Default = 1.
model, agent_data, transmission_network, social_contact_matrix, epidemic_summary = Run_Model!(model) # the social_contact_matrix returned is only the upper half. To reconstruct entire matrix use decompact_adjacency_matrix(filename)
