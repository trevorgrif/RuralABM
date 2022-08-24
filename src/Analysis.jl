#============================================================
---------------------- Main Functions -----------------------
============================================================#

function get_epidemic_data(model, AgentData)
    AgentDataDaily = parse_to_daily!(AgentData)

    # Compute Statistics
    infected_total = AgentData[end,2] + AgentData[end,3] + (model.init_pop_size - AgentData[end,4]) # Current_Infected + Recovered + Dead
    recovered_total = AgentData[end,3]
    infected_max, peak_day = findmax(AgentDataDaily.Symptomatic)
    peak_day -= 1

    recovered_masked = 0
    recovered_vaccinated = 0
    recovered_MandV = 0
    for agent in allagents(model)
        if agent.status == :R
            if agent.will_mask[1] == true
                recovered_masked += 1
            end
            if agent.vaccinated == true
                recovered_vaccinated += 1
            end
            if agent.vaccinated == true && agent.will_mask[1] == true
                recovered_MandV += 1
            end
        end
    end

    EpidemicDataDF = DataFrame(InfectedTotal = infected_total,  InfectedMax = infected_max, PeakDay = peak_day, RecoveredTotal = recovered_total, RecoveredMasked = recovered_masked, RecoveredVaccinated = recovered_vaccinated, RecoveredMandV = recovered_MandV)

    return EpidemicDataDF
end

function compute_spectral_radius(contact_matrix)
    eigenvalues = eigen(contact_matrix).values
    eigenvalues_normalized = broadcast(abs, eigenvalues)
    spectral_radius = maximum(eigenvalues_normalized)
    return spectral_radius
end

function compute_spectral_radius_from_filename(filename, norm_days)
    adjacency_matrix = decompact_adjacency_matrix(filename)
    adjacency_matrix /= norm_days*12
    return compute_spectral_radius(adjacency_matrix)
end

#============================================================
--------------------- Helper Functions ----------------------
============================================================#

function parse_to_daily!(AgentData)
    day_data = AgentData[1:12:end, :]
    select!(day_data,:step => (x -> x/12), :count_symptomatic, :count_recovered, :count_pop_size)
    rename!(day_data, Dict(:step_function => "Day", :count_symptomatic => "Symptomatic", :count_recovered => "Recovered", :count_pop_size => "Population"))
    return day_data
end
