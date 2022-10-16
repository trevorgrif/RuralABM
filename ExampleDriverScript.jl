#=
Example driver script for running the RuralABM model
=#

# Load RuralABM
using Pkg
Pkg.activate(".")
using RuralABM

# Initialize town with household, work, and school assignments
model_init, townDataSummaryDF, businessStructureDF, houseStructureDF = Construct_Town("data/example_towns/small_town/population.csv", "data/example_towns/small_town/businesses.csv")

# Run the model without any contagion to converge the social network
length_to_run_in_days = 15
Run_Model!(model_init; duration = length_to_run_in_days)

# Apply vaccination and masking behaviors to certain age ranges
portion_will_mask = 0.1
portion_vaxed = 0.1

# Distribute mask and vax randomly
mask_id_arr_rand = Get_Portion_Random(model_init, portion_will_mask, [(x)->x.age >= 2])
vaccinated_id_arr_rand = Get_Portion_Random(model_init, portion_vaxed, [(x)-> x.age > 4], [1.0])

# Distribute mask and vax with Watts threshold model
mask_id_arr_watts = Get_Portion_Watts(model_init, portion_will_mask)
vaccinated_id_arr_watts = Get_Portion_Watts(model_init, portion_vaxed)

Update_Agents_Attribute!(model_init, mask_id_arr_watts, :will_mask, [true, true, true])
Update_Agents_Attribute!(model_init, vaccinated_id_arr_watts, :status, :V)
Update_Agents_Attribute!(model_init, vaccinated_id_arr_watts, :vaccinated, true)

# Run the model with contagion until the count of infected agents is zero
Seed_Contagion!(model_init) # set parameter seed_num = x for x seedings. Default = 1.
model_result, agent_data, transmission_network, social_contact_matrix, epidemic_summary = Run_Model!(model_init) # the social_contact_matrix returned is only the upper half. To reconstruct entire matrix use decompact_adjacency_matrix(filename)

# Extract Adjacency Matrix
#using DataFrames, CSV
#arr = RuralABM.get_adjacency_matrix(model)
#df = DataFrame([eachcol(arr)...], :auto, copycols=false)
#CSV.write("SCM.csv", df, header = false)
