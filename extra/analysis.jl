using Graphs

#######################################
#   Age Structured Contact Matrices   #
#######################################

function plot_age_structured_contact(NetworkID, connection)
    data = compute_age_structured_contact_matrix(NetworkID, connection)
    data = unstack(data, :ContactGroup, :Weight)
    # data = data_unstacked[1:end,2:end]

    data = select(data, 
        AsTable(Cols("00-04")) => sum, 
        AsTable(Cols("05-09")) => sum, 
        AsTable(Cols("10-14")) => sum, 
        AsTable(Cols("15-17","18-19")) => sum, 
        AsTable(Cols("20-20","21-21","22-24","25-29")) => sum, 
        AsTable(Cols("30-34","35-39")) => sum, 
        AsTable(Cols("40-44","45-49")) => sum, 
        AsTable(Cols("50-54","55-59")) => sum, 
        AsTable(Cols("60-61","62-64","65-66","67-69")) => sum, 
        AsTable(Cols("70-74","75-79","80-84","85-NA")) => sum
        )

    col1 = sum.(eachcol(data[1:1,1:end]))
    col2 = sum.(eachcol(data[2:2,1:end]))
    col3 = sum.(eachcol(data[3:3,1:end]))
    col4 = sum.(eachcol(data[4:5,1:end]))
    col5 = sum.(eachcol(data[6:9,1:end]))
    col6 = sum.(eachcol(data[10:11,1:end]))
    col7 = sum.(eachcol(data[12:13,1:end]))
    col8 = sum.(eachcol(data[14:15,1:end]))
    col9 = sum.(eachcol(data[16:19,1:end]))
    col10 = sum.(eachcol(data[20:end,1:end]))

    data = DataFrame(
        A = col1,
        B = col2,
        C = col3,
        D = col4,
        E = col5,
        F = col6,
        G = col7,
        H = col8,
        I = col9,
        J = col10
        )
    
    AgeData = [
        "0 to 4",
        "5 to 9",
        "10 to 14",
        "15 to 19",
        "20 to 29",
        "30 to 39", 
        "40 to 49", 
        "50 to 59", 
        "60 to 69", 
        "70+"
        ]

    @show data
    plot(
        heatmap(
            x = AgeData,
            y = AgeData,
            z = Matrix(data),
        ),
        Layout(
            xaxis_title="Age",
            yaxis_title="Age of Contact",
            title="Age Structured Contacts: Network $NetworkID"
        )
    )
end

function age_structured_contact(PopulationID, con)
    query = """
    SELECT AgeRangeID, string_agg(AgentID, ',')  AS AgentList
    FROM PopulationLoad 
    WHERE PopulationID = $PopulationID
    GROUP BY AgeRangeID
    ORDER BY AgeRangeID
    """
    return DataFrame(run_query(query,con))
end

function cross_age_range_contacts(AgentIDs1::Vector{Int64}, AgentIDs2::Vector{Int64}, NetworkID, con)
    query = """
    SELECT SUM(Weight) AS TotalContacts 
    FROM NetworkSCMLoad 
    WHERE (NetworkID = $NetworkID 
    AND Agent1 IN ($(string(AgentIDs1)[2:end-1])) 
    AND Agent2 IN ($(string(AgentIDs2)[2:end-1]))
    )
    OR (NetworkID = $NetworkID
    AND Agent2 IN ($(string(AgentIDs1)[2:end-1])) 
    AND Agent1 IN ($(string(AgentIDs2)[2:end-1]))
    )
    """
    result = DataFrame(run_query(query, con))[1,1]
    
    if typeof(result) == Missing
        return 0
    end
    return result 
end

function compute_age_structured_contact_matrix(NetworkID, connection)
    AgeRangeDF = age_structured_contact(1,connection)

    Data::Matrix{Int64} = zeros(size(AgeRangeDF,1), size(AgeRangeDF,1))
    AgeRangeItr = 1
    DataDF = DataFrame(AgeGroup = String[], ContactGroup = String[], Weight = Int64[])
    for AgeRange in eachrow(AgeRangeDF)
        SubRangeItr = 1
        for SubRange in eachrow(AgeRangeDF)
            # Compute the contacts AgeRange have with SubRange
            AgeRangeVector::Vector{Int64} = convert_to_vector(AgeRange[2])
            SubRangeVector::Vector{Int64} = convert_to_vector(SubRange[2]) 
            Weight = cross_age_range_contacts(AgeRangeVector, SubRangeVector, NetworkID, connection)
            Data[AgeRangeItr, SubRangeItr] = Weight
            SubRangeItr += 1
            append!(DataDF, DataFrame(AgeGroup = [AgeRange[1]], ContactGroup = [SubRange[1]], Weight = [Weight]))
        end 
        AgeRangeItr += 1
    end
    return DataDF
    
    return data
end

function convert_to_vector(List)
    return parse.(Int64, split(List, ","))
end

############################
#    Summary Statistics    #
############################

function ratio_infection_deaths(connection)
    query = """
        SELECT MaskPortion, VaxPortion, AVG(CAST(InfectedTotal AS DECIMAL) / (386 - InfectedTotal + RecoveredTotal)) AS RatioInfectionDeaths 
        FROM EpidemicDim
        GROUP BY MaskPortion, VaxPortion 
    """
    run_query(query, connection)
end

function ComputeSummaryStats(Population, con)

    OutbreakThreshold = 0.1*Population

    query = """
    WITH MaskedAndVaxedAgents AS (
        SELECT 
            AgentLoad.BehaviorID,
            AgentLoad.AgentID
        FROM AgentLoad
        WHERE IsMasking = 1
        AND IsVaxed = 1
    ),
    InfectedAndProtectedAgents AS (
        SELECT 
            NetworkDim.NetworkID,
            BehaviorDim.BehaviorID,
            EpidemicDim.EpidemicID,
            COUNT(TransmissionLoad.AgentID) AS ProtectedAndInfectedCount
        FROM BehaviorDim
        JOIN EpidemicDim
        ON EpidemicDim.BehaviorID = BehaviorDim.BehaviorID
        JOIN TransmissionLoad
        ON TransmissionLoad.EpidemicID = EpidemicDim.EpidemicID
        JOIN NetworkDim
        ON NetworkDim.NetworkID = BehaviorDim.NetworkID
        WHERE TransmissionLoad.AgentID IN ( 
            SELECT AgentID 
            FROM MaskedAndVaxedAgents
            WHERE MaskedAndVaxedAgents.BehaviorID = BehaviorDim.BehaviorID
            )
        GROUP BY  NetworkDim.NetworkID, BehaviorDim.BehaviorID, EpidemicDim.EpidemicID
    ),
    AggregateInfectedAndProtectedCount AS (
        SELECT 
            InfectedAndProtectedAgents.NetworkID,
            InfectedAndProtectedAgents.BehaviorID,
            AVG(ProtectedAndInfectedCount) AS AverageMaskedVaxedInfectedCount
        FROM InfectedAndProtectedAgents
        GROUP BY InfectedAndProtectedAgents.BehaviorID, InfectedAndProtectedAgents.NetworkID
    ),
    MaskVaxCounts AS (
        SELECT 
            BehaviorID,
            SUM(IsMasking) AS IsMaskingCount,
            SUM(IsVaxed) AS IsVaxedCount,
            SUM(CASE WHEN IsVaxed = 1 THEN IsMasking ELSE 0 END) AS IsMaskingAndVaxed
        FROM AgentLoad
        GROUP BY BehaviorID
    ),
    TownData AS (
        SELECT 
            lpad(BehaviorDim.NetworkID, 2, '0') AS NetworkID, 
            TownDim.MaskDistributionType,
            TownDim.VaxDistributionType,
            MaskVaxDim.MaskPortion, 
            MaskVaxDim.VaxPortion,
            IsMaskingCount,
            IsVaxedCount,
            IsMaskingAndVaxed AS IsMaskingVaxedCount,
            CASE WHEN AverageMaskedVaxedInfectedCount IS NULL THEN 0 ELSE AverageMaskedVaxedInfectedCount END AS AverageMaskedVaxedInfectedCount,
            InfectedTotal,
            InfectedMax,
            PeakDay,
            RecoveredTotal
        FROM TownDim
        JOIN NetworkDim
        ON NetworkDim.TownID = TownDim.TownID
        JOIN BehaviorDim        
        ON NetworkDim.NetworkID = BehaviorDim.NetworkID
        JOIN MaskVaxDim
        ON BehaviorDim.MaskVaxID = MaskVaxDim.MaskVaxID
        JOIN MaskVaxCounts
        ON MaskVaxCounts.BehaviorID = BehaviorDim.BehaviorID
        JOIN EpidemicDim
        ON BehaviorDim.BehaviorID = EpidemicDim.BehaviorID
        LEFT JOIN AggregateInfectedAndProtectedCount
        ON AggregateInfectedAndProtectedCount.BehaviorID = BehaviorDim.BehaviorID
    )
    SELECT 
        NetworkID, 
        MaskDistributionType,
        VaxDistributionType,
        MaskPortion, 
        VaxPortion,
        IsMaskingCount,
        IsVaxedCount,
        IsMaskingVaxedCount,
        AverageMaskedVaxedInfectedCount,
        AverageMaskedVaxedInfectedCount/IsMaskingVaxedCount AS ProbabilityOfInfectionWhenProtected,
        AVG(InfectedTotal) AS AverageInfectedTotal, 
        AVG(InfectedTotal)/$Population AS AverageInfectedPercentage,
        var_samp(InfectedTotal) AS VarianceInfectedTotal, 
        AVG(InfectedMax) AS AverageInfectedMax, 
        var_samp(InfectedMax) AS VarianceInfectedMax,
        AVG(PeakDay) AS AveragePeakDay, 
        var_samp(PeakDay)  As VariancePeakDay,
        AVG(CAST(InfectedTotal AS DECIMAL) / (386 - InfectedTotal + RecoveredTotal)) AS RatioInfectionDeaths 
    FROM TownData
    WHERE InfectedTotal > $(OutbreakThreshold)
    GROUP BY NetworkID, MaskPortion, VaxPortion, MaskDistributionType, VaxDistributionType, IsMaskingCount, IsVaxedCount, IsMaskingVaxedCount, AverageMaskedVaxedInfectedCount, ProbabilityOfInfectionWhenProtected
    ORDER BY MaskPortion, VaxPortion, NetworkID
    """
    #run_query(query, con)
    CSV.write("StatsDF.csv", run_query(query, con) |> DataFrame)
end

function Compute_Global_Clustering_Coefficient(connection)
    # Iterate over Network SCM
    NetworkIDs = [1,2,3,4,5,6,7,8,9,10]
    GlobalClusteringCoefficients = []
    for NetworkID in NetworkIDs
        # Extract Network Data into Graphs.jl Graph object
        query = """
        SELECT Agent1, Agent2, Weight
        FROM NetworkSCMLoad
        WHERE NetworkID = $NetworkID
        """
        NetworkSCMLoad = run_query(query, connection) |> DataFrame
        

        # Compute global clustering coefficient
        GlobalClusteringCoefficient =  global_clustering_coefficient(NetworkSCM)

        # Load into array of results
        append!(GlobalClusteringCoefficients, GlobalClusteringCoefficient)
    end

    return GlobalClusteringCoefficients
end

"""
Plot the population by age range as a bar graph
"""
function plot_population_distribution(PopulationID, connection)
    query = """
    SELECT 
        replace(AgeRangeID, '-', ' to ')AS AgeRangeID, 
        COUNT(*) AS BinSize 
    FROM PopulationLoad 
    WHERE PopulationID = $PopulationID 
    GROUP BY AgeRangeID 
    ORDER BY AgeRangeID
    """
    data = run_query(query, connection) |> DataFrame

    col1 = sum.(eachcol(data[1:1,2:end]))
    col2 = sum.(eachcol(data[2:2,2:end]))
    col3 = sum.(eachcol(data[3:3,2:end]))
    col4 = sum.(eachcol(data[4:5,2:end]))
    col5 = sum.(eachcol(data[6:9,2:end]))
    col6 = sum.(eachcol(data[10:11,2:end]))
    col7 = sum.(eachcol(data[12:13,2:end]))
    col8 = sum.(eachcol(data[14:15,2:end]))
    col9 = sum.(eachcol(data[16:19,2:end]))
    col10 = sum.(eachcol(data[20:end,2:end]))

    data = DataFrame(AgeRangeID = String[], BinSize = Int64[])
    append!(data, DataFrame(AgeRangeID = "00 to 04", BinSize = col1[1]))
    append!(data, DataFrame(AgeRangeID = "05 to 09", BinSize = col2[1]))
    append!(data, DataFrame(AgeRangeID = "10 to 14", BinSize = col3[1]))
    append!(data, DataFrame(AgeRangeID = "15 to 19", BinSize = col4[1]))
    append!(data, DataFrame(AgeRangeID = "20 to 29", BinSize = col5[1]))
    append!(data, DataFrame(AgeRangeID = "30 to 39", BinSize = col6[1]))
    append!(data, DataFrame(AgeRangeID = "40 to 49", BinSize = col7[1]))
    append!(data, DataFrame(AgeRangeID = "50 to 59", BinSize = col8[1]))
    append!(data, DataFrame(AgeRangeID = "60 to 69", BinSize = col9[1]))
    append!(data, DataFrame(AgeRangeID = "70+", BinSize = col10[1]))

    @show data

    plot(
        data,
        x = :AgeRangeID,
        y = :BinSize,
        kind = "bar"
    )

end

###################
#    Thickness    #
###################
using Ripserer, Graphs

function compute_thickness(NetworkID, connection)
    PersistenceDiagram = compute_persistence_diagram(NetworkID, connection)

    H1Epsilons = Tables.matrix(PersistenceDiagram[2])
    H2Epsilons = Tables.matrix(PersistenceDiagram[3])

    SignificantEpsilons::Vector{Float64} = vcat(H1Epsilons[:,1], H1Epsilons[:, 2], H2Epsilons[:,1], H2Epsilons[:, 2])
    SignificantEpsilons = SignificantEpsilons |> unique |> sort

    Ranks = DataFrame(dist = SignificantEpsilons, h1 = zeros(length(SignificantEpsilons)), h2 = zeros(length(SignificantEpsilons)), sum = zeros(length(SignificantEpsilons)))
    
    # Iterate over h1s
    for row in eachrow(H1Epsilons)
        Ranks[Ranks.dist .>= row[1] .&& Ranks.dist .< row[2], [2,4]] .+= 1.0
    end

    # Iterate over h2s
    for row in eachrow(H2Epsilons)
        Ranks[Ranks.dist .>= row[1] .&& Ranks.dist .< row[2], [3,4]] .+= 1.0
    end
    
    RanksFiltered = Ranks[Ranks.sum .> 3.0, :]
    RanksComputed = select(RanksFiltered, :dist, :h1, :h2, :sum, [:h1, :h2, :sum] => ((h1, h2, sum) -> (h2 .- h1)./sum) => :tau)
    
    return RanksComputed
end

function compute_persistence_diagram(NetworkID, connection)
    SCM = network_SCM_to_matrix(386, NetworkID, connection)
    SCM = 1.0 ./ SCM
    DistanceMatrix = floyd_warshall_shortest_paths(Graph(SCM), SCM).dists
    PersistenceDiagram = ripserer(DistanceMatrix; dim_max=2)
    
    return PersistenceDiagram
end

function network_SCM_to_matrix(PopulationSize, NetworkID, connection)
    SCMMatrix = zeros(PopulationSize, PopulationSize)
    query = """
        SELECT Agent1, Agent2, Weight 
        FROM NetworkSCMLoad
        WHERE NetworkID = $NetworkID
        ORDER BY Agent1, Agent2
        """
    SCMNetwork = DataFrame(run_query(query, connection))
    for RowIdx in 1:PopulationSize
        SCMMatrix[RowIdx, (RowIdx+1):end] =  SCMNetwork[SCMNetwork.Agent1 .== RowIdx, 3:3] |> Matrix |> transpose
    end
    return SCMMatrix + transpose(SCMMatrix)
end


function foo(connection)
    for i in 1:5:36
        thick = compute_thickness(i, connection)
        savefig(plot(thick[:,1], thick[:,5]), "Thickness_$(lpad(i,2,"0")).png")
    end
end