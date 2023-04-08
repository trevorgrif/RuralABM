using DuckDB
using DataFrames
using StatsBase
using PlotlyJS

#====================#
# Connection Methods #
#====================#

function run_query(query, connection)
    DBInterface.execute(connection, query)
end

function create_con()
    return DBInterface.connect(DuckDB.DB, "data/GDWLND.duckdb")
end

function close_con(connection)
    DBInterface.close(connection)
end

#=========================#
# Epidemic Level Analysis #
#=========================#

function get_epidemic_level_data(connection)
    query = """
    SELECT DISTINCT
        EpidemicDim.EpidemicID,
        BehaviorDim.BehaviorID,
        NetworkDim.NetworkID,
        TownDim.TownID,
        BehaviorDim.MaskVaxID,
        MaskVaxDim.MaskPortion,
        MaskVaxDim.VaxPortion,
        EpidemicDim.InfectedTotal,
        EpidemicDim.InfectedMax,
        EpidemicDim.PeakDay,
        EpidemicDim.RecoveredTotal,
        EpidemicDim.RecoveredMasked,
        EpidemicDim.RecoveredVaccinated,
        EpidemicDim.RecoveredMaskAndVax,
        ProtectedTotals.MaskedAgentTotal,
        ProtectedTotals.VaxedAgentTotal,
        ProtectedTotals.MaskAndVaxAgentTotal,
        InfectedTotals.InfectedMaskedAgentTotal,
        InfectedTotals.InfectedVaxedAgentTotal,
        InfectedTotals.InfectedMaskAndVaxAgentTotal,
        CASE WHEN ProtectedTotals.MaskAndVaxAgentTotal = 0 THEN 0 ELSE CAST(InfectedTotals.InfectedMaskAndVaxAgentTotal AS DECIMAL) / CAST(ProtectedTotals.MaskAndVaxAgentTotal AS DECIMAL) END AS InfectedMaskAndVaxAgentProbability,
    FROM EpidemicDim
    LEFT JOIN BehaviorDim ON BehaviorDim.BehaviorID = EpidemicDim.BehaviorID
    LEFT JOIN NetworkDim ON NetworkDim.NetworkID = BehaviorDim.NetworkID
    LEFT JOIN TownDim ON TownDim.TownID = NetworkDim.TownID
    LEFT JOIN MaskVaxDim ON MaskVaxDim.MaskVaxID = BehaviorDim.MaskVaxID
    LEFT JOIN (
        -- Get the total number of infected agents in each protected class for each epidemic
        SELECT
            EpidemicDim.BehaviorID,
            EpidemicDim.EpidemicID,
            SUM(CASE WHEN TransmissionLoad.AgentID in (
                SELECT AgentLoad.AgentID
                FROM AgentLoad
                WHERE AgentLoad.IsMasking = 1
                AND AgentLoad.BehaviorID = EpidemicDim.BehaviorID
            ) THEN 1 ELSE 0 END) AS InfectedMaskedAgentTotal,
            SUM(CASE WHEN TransmissionLoad.AgentID in (
                SELECT AgentLoad.AgentID
                FROM AgentLoad
                WHERE AgentLoad.IsVaxed = 1
                AND AgentLoad.BehaviorID = EpidemicDim.BehaviorID
            ) THEN 1 ELSE 0 END) AS InfectedVaxedAgentTotal,
            SUM(CASE WHEN TransmissionLoad.AgentID in (
                SELECT AgentLoad.AgentID
                FROM AgentLoad
                WHERE AgentLoad.IsMasking = 1 AND AgentLoad.IsVaxed = 1
                AND AgentLoad.BehaviorID = EpidemicDim.BehaviorID
            ) THEN 1 ELSE 0 END) AS InfectedMaskAndVaxAgentTotal
        FROM EpidemicDim
        LEFT JOIN TransmissionLoad
        ON TransmissionLoad.EpidemicID = EpidemicDim.EpidemicID
        GROUP BY EpidemicDim.BehaviorID, EpidemicDim.EpidemicID
    ) InfectedTotals
    ON InfectedTotals.EpidemicID = EpidemicDim.EpidemicID
    LEFT JOIN (
        -- Get the total number of agents in each protected class
        SELECT DISTINCT
            AgentLoad.BehaviorID,
            SUM(CASE WHEN AgentLoad.IsMasking = 1 THEN 1 ELSE 0 END) AS MaskedAgentTotal,
            SUM(CASE WHEN AgentLoad.IsVaxed = 1 THEN 1 ELSE 0 END) AS VaxedAgentTotal,
            SUM(CASE WHEN AgentLoad.IsVaxed = 1 THEN AgentLoad.IsMasking ELSE 0 END) AS MaskAndVaxAgentTotal
        FROM AgentLoad
        GROUP BY AgentLoad.BehaviorID
    ) ProtectedTotals
    ON ProtectedTotals.BehaviorID = InfectedTotals.BehaviorID
    ORDER BY 1,2
    """
    Result = run_query(query, connection) |> DataFrame 
end

"""
    epidemic_level_computed_statistics(data)

Function to compute the average of InfectedMaskAndVaxAgentProbability aggregated by MaskVaxID

`data` should be subset of the data returned by `get_epidemic_level_data`.
"""
function epidemic_level_computed_statistics(data)
    # Compute the average of InfectedMaskAndVaxAgentProbability aggregated by MaskVaxID
    avg_infected_mask_and_vax_agent_probability = unique(combine(groupby(data,[:TownID, :NetworkID, :MaskPortion, :VaxPortion]), [:MaskPortion, :VaxPortion] => ((m,v) -> 10*m + v ),:InfectedMaskAndVaxAgentProbability => mean))
    return avg_infected_mask_and_vax_agent_probability
end

#==============#
# Watts Effect #
#==============#

"""
    watts_statistical_test(target_distribution = 0)

Perform a statistical t-test (Welch's T-Test) to determine if the average of InfectedMaskAndVaxAgentProbability is significantly different between town builds at the given target distribution.

Returns an array of test results between all four town builds.

# Note: target_distribution = 10 * mask_portion + vax_portion
"""
function watts_statistical_test(target_distribution = 0)
    data = create_con() |> get_epidemic_level_data
    data = data[data.TownID .< 5, :]

    data_1 = data[data.TownID .== 1, :]
    data_2 = data[data.TownID .== 2, :]
    data_3 = data[data.TownID .== 3, :]
    data_4 = data[data.TownID .== 4, :]

    select!(data_1, :TownID, [:MaskPortion, :VaxPortion] => ((m,v) -> 10*m+v), :InfectedMaskAndVaxAgentProbability)
    select!(data_2, :TownID, [:MaskPortion, :VaxPortion] => ((m,v) -> 10*m+v), :InfectedMaskAndVaxAgentProbability)
    select!(data_3, :TownID, [:MaskPortion, :VaxPortion] => ((m,v) -> 10*m+v), :InfectedMaskAndVaxAgentProbability)
    select!(data_4, :TownID, [:MaskPortion, :VaxPortion] => ((m,v) -> 10*m+v), :InfectedMaskAndVaxAgentProbability)

    ProtectedInfectionProbability_1 = convert.(Float64, data_1[data_1.MaskPortion_VaxPortion_function .== target_distribution, :InfectedMaskAndVaxAgentProbability])
    ProtectedInfectionProbability_2 = convert.(Float64, data_2[data_2.MaskPortion_VaxPortion_function .== target_distribution, :InfectedMaskAndVaxAgentProbability])
    ProtectedInfectionProbability_3 = convert.(Float64, data_3[data_3.MaskPortion_VaxPortion_function .== target_distribution, :InfectedMaskAndVaxAgentProbability])
    ProtectedInfectionProbability_4 = convert.(Float64, data_4[data_4.MaskPortion_VaxPortion_function .== target_distribution, :InfectedMaskAndVaxAgentProbability])

    Results = []
    append!(Results, [UnequalVarianceTTest(ProtectedInfectionProbability_1, ProtectedInfectionProbability_2)])
    append!(Results, [UnequalVarianceTTest(ProtectedInfectionProbability_1, ProtectedInfectionProbability_3)])
    append!(Results, [UnequalVarianceTTest(ProtectedInfectionProbability_1, ProtectedInfectionProbability_4)])
    append!(Results, [UnequalVarianceTTest(ProtectedInfectionProbability_2, ProtectedInfectionProbability_3)])
    append!(Results, [UnequalVarianceTTest(ProtectedInfectionProbability_2, ProtectedInfectionProbability_4)])
    append!(Results, [UnequalVarianceTTest(ProtectedInfectionProbability_3, ProtectedInfectionProbability_4)])

    return Results

end

"""
    plot_probability_infection_protected()

Plot the average of InfectedMaskAndVaxAgentProbability aggregated by MaskVaxID for each town build. Evidence of `Watts Effect`.
"""
function plot_probability_infection_protected()
    data = create_con() |> get_epidemic_level_data
    data = data[data.TownID .< 5, :]
    data = epidemic_level_computed_statistics(data)

    # Filter out 0% infection probability
    data = data[data.InfectedMaskAndVaxAgentProbability_mean .> 0, :]

    data_1 = data[data.TownID .== 1, :]
    data_2 = data[data.TownID .== 2, :]
    data_3 = data[data.TownID .== 3, :]
    data_4 = data[data.TownID .== 4, :]

    plot(
        [
            scatter(
            data_1, 
            name = "Town 1 (R,R)",
            x=:MaskPortion_VaxPortion_function, 
            y=:InfectedMaskAndVaxAgentProbability_mean, 
            facet_color=:TownID,
            mode="markers"
            ),
            scatter(
            data_2,
            name = "Town 2 (W,W)",
            x=:MaskPortion_VaxPortion_function,
            y=:InfectedMaskAndVaxAgentProbability_mean,
            facet_color=:TownID,
            mode="markers"
            ),
            scatter(
            data_3,
            name = "Town 3 (W,R)",
            x=:MaskPortion_VaxPortion_function,
            y=:InfectedMaskAndVaxAgentProbability_mean,
            facet_color=:TownID,
            mode="markers"
            ),
            scatter(
            data_4,
            name = "Town 4 (R,W)",
            x=:MaskPortion_VaxPortion_function,
            y=:InfectedMaskAndVaxAgentProbability_mean,
            facet_color=:TownID,
            mode="markers"
            )
        ],
        Layout(
            legend = attr(
                y = 1.02,
                x = 1,
                yanchor="bottom",
                xanchor="right",
                orientation="h"
            ),
            title="Probability of Infection for Protected Agents",
            xaxis_title="Masking and Vaccination Distribution (10*m+v)",
            yaxis_title="Probability of Infection for Protected Agents",
            legend_title="TownID",
            legend_orientation="h"
        )
    )
end