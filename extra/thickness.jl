using Ripserer

############################
# Epidemic Level Thickness #
############################

"""
    thickness_metrics(connection)

Queries thickness mean and variance along with meta-data on the epidemic.
"""
function thickness_metrics(connection; SaveResults = false, Population = 386)
    OutbreakThreshold = 0.20*Population
    FailedSeedingThreshold = 0.00*Population

    # Compute mean and variance of thickness for each mask/vask level agg. by network level
    query = """
    WITH ThicknessTable AS (
        SELECT 
            NetworkDim.NetworkID,
            BehaviorDim.BehaviorID,
            PersistenceLoad.EpidemicID,
            EpidemicDim.InfectedTotal,
            EpidemicDim.InfectedMax,
            EpidemicDim.PeakDay,
            BehaviorDim.MaskVaxID,
            PersistenceLoad.Distance,
            PersistenceLoad.H0Count,
            PersistenceLoad.H1Count,
            PersistenceLoad.H2Count,
            (CAST(H2Count AS DECIMAL) - CAST(H1Count AS DECIMAL))/(H0Count + H1Count + H2Count) AS Thickness
        FROM PersistenceLoad
        JOIN EpidemicDim ON EpidemicDim.EpidemicID = PersistenceLoad.EpidemicID
        JOIN BehaviorDim ON BehaviorDim.BehaviorID = EpidemicDim.BehaviorID
        JOIN NetworkDim ON NetworkDim.NetworkID = BehaviorDim.NetworkID
    )
    SELECT
        ThicknessTable.NetworkID,
        ThicknessTable.BehaviorID,
        ThicknessTable.EpidemicID,
        ThicknessTable.MaskVaxID,
        ThicknessTable.InfectedTotal,
        ThicknessTable.InfectedMax,
        ThicknessTable.PeakDay,
        CASE WHEN ThicknessTable.InfectedTotal <= $FailedSeedingThreshold THEN -1 WHEN ThicknessTable.InfectedTotal <= $OutbreakThreshold THEN 0 ELSE 1 END AS Outbreak,
        AVG(ThicknessTable.Thickness) AS AverageThickness,
        VAR_POP(ThicknessTable.Thickness) AS PopulationVarianceThickness
    FROM ThicknessTable
    GROUP BY ThicknessTable.NetworkID, ThicknessTable.BehaviorID, ThicknessTable.EpidemicID, ThicknessTable.InfectedTotal, ThicknessTable.MaskVaxID, ThicknessTable.InfectedMax, ThicknessTable.PeakDay
    """
    ResultDF = run_query(query, connection) |> DataFrame    
    SaveResults && CSV.write("thickness_metrics.csv", ResultDF)
    return ResultDF
end

"""
import_thickness(TownID, EpidemicIDLeft, EpidemicIDRight, BatchSize, connection)

Computes the feature count at each significant epsilon for the given range of epidemic ids and loads the results into PersistenceLoad. Supports multi-threading.
"""
function import_thickness(TownID, EpidemicIDLeft, EpidemicIDRight, BatchSize, connection)
    query = """
    WITH x AS (
        SELECT n 
        FROM (VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9)) v(n)
        ),
    y AS (
        SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS ID
        FROM x ones, x tens, x hundreds, x thousands, x tenthousands, x hundreadthousand
        )
    SELECT *
    FROM EpidemicSCMLoad_$TownID
    JOIN y
    ON y.ID = EpidemicSCMLoad_$TownID.EpidemicID
    WHERE y.ID not in (SELECT DISTINCT EpidemicID FROM PersistenceLoad)
    AND EpidemicID >= $EpidemicIDLeft
    AND EpidemicID <= $EpidemicIDRight
    ORDER BY EpidemicID
    """
    ResultDF = run_query(query, connection) |> DataFrame

    # Partition the dataframe for batched multiprocessing
    RowPartitions = Iterators.partition(1:nrow(ResultDF), BatchSize) |> collect
    for Partition in RowPartitions
        @show ResultDF[Partition, 1:1]
        import_thickness_batch(ResultDF[Partition, :], connection)
        GC.gc()
    end 
end

"""
    import_thickness_batch(ResultDF, connection)

Helper function to `import_thickness`. Enables multi-threading.
"""
function import_thickness_batch(ResultDF, connection)
    # Parallel Persistence Computations
    FeatureResults = Vector{Any}(missing, nrow(ResultDF))

    # @threads version
    Threads.@threads for i in 1:nrow(ResultDF)
        row = ResultDF[i, 1:end]
        feature_count_mt(row, FeatureResults, i)
    end
    
    # Insert into DataTable
    for set in FeatureResults
        RowPartitions = Iterators.partition(1:size(set,1), 100) |> collect
        for Partition in RowPartitions
            PersistenceLoadAppender = DuckDB.Appender(connection, "PersistenceLoad")
            for row in eachrow(set[Partition, :])
                DuckDB.append(PersistenceLoadAppender, row[1])
                DuckDB.append(PersistenceLoadAppender, row[2])
                DuckDB.append(PersistenceLoadAppender, row[3])
                DuckDB.append(PersistenceLoadAppender, row[4])
                DuckDB.append(PersistenceLoadAppender, row[5])
                DuckDB.end_row(PersistenceLoadAppender)
            end
            DuckDB.close(PersistenceLoadAppender)
        end
    end
end

"""
    feature_count_mt(row, FeatureResults, i)

Calls epidemic_persistence_diagrame and feature_count on current thread and stores results in FeatureResults[i].
"""
function feature_count_mt(row, FeatureResults, i)
    FeatureCountDF = epidemic_persistence_diagram(386, row[2]) |> feature_count
    FeatureCount = size(FeatureCountDF, 1)
    EpidemicIDArray = zeros(FeatureCount)
    EpidemicIDArray .= row[1]
    FeatureResults[i] = hcat(EpidemicIDArray, FeatureCountDF)
    return
end

"""
    epidemic_persistence_diagram(PopulationSize, SCM)

Use ripserer.jl to compute the Vietoris-Rips complex of the social contact matrix using the floyd warshall shortest path algorithm to compute the distance matrix.
"""
function epidemic_persistence_diagram(PopulationSize, SCM)
    SCM = epidemic_scm_to_matrix(PopulationSize, SCM)
    SCM = 1.0 ./ SCM
    DistanceMatrix = floyd_warshall_shortest_paths(Graph(SCM), SCM).dists
    PersistenceDiagram = ripserer(DistanceMatrix; dim_max=2, threshold=0.15)
    
    return PersistenceDiagram
end

"""
    epidemic_scm_to_matrix(PopulationSize, SCM)

Unpacks SCM from the upper-half form to the full social contact matrix.
"""
function epidemic_scm_to_matrix(PopulationSize, SCM)
    SCMMatrix = zeros(PopulationSize, PopulationSize)
    SCM = convert_to_vector(SCM)
    
    Shift = PopulationSize-2
    StartIdx = 1
    for RowIdx in 1:(PopulationSize-1)
        SCMMatrix[(RowIdx+1):end, RowIdx] =  SCM[StartIdx:(StartIdx + Shift)] 
        StartIdx += Shift + 1
        Shift -= 1
    end
    return SCMMatrix + transpose(SCMMatrix)
end

"""
    feature_count(PersistenceDiagrame)

Returns a matrix object with the column order: :dist, :h0, :h1, :h2
"""
function feature_count(PersistenceDiagram)
    H0Epsilons = Tables.matrix(PersistenceDiagram[1])
    H1Epsilons = Tables.matrix(PersistenceDiagram[2])
    H2Epsilons = Tables.matrix(PersistenceDiagram[3])

    SignificantEpsilons::Vector{Float64} = vcat(H0Epsilons[:,1], H0Epsilons[:, 2], H1Epsilons[:,1], H1Epsilons[:, 2], H2Epsilons[:,1], H2Epsilons[:, 2])
    SignificantEpsilons = SignificantEpsilons |> unique |> sort

    Ranks = DataFrame(dist = SignificantEpsilons, h0 = zeros(length(SignificantEpsilons)), h1 = zeros(length(SignificantEpsilons)), h2 = zeros(length(SignificantEpsilons)))

    # Iterate over h0s
    for row in eachrow(H0Epsilons)
        Ranks[Ranks.dist .>= row[1] .&& Ranks.dist .< row[2], [2]] .+= 1.0
    end

    # Iterate over h1s
    for row in eachrow(H1Epsilons)
        Ranks[Ranks.dist .>= row[1] .&& Ranks.dist .< row[2], [3]] .+= 1.0
    end

    # Iterate over h2s
    for row in eachrow(H2Epsilons)
        Ranks[Ranks.dist .>= row[1] .&& Ranks.dist .< row[2], [4]] .+= 1.0
    end
    
    RanksComputed = select(Ranks, :dist, :h0, :h1, :h2)

    return Matrix(RanksComputed)
end

"""
    convert_to_vector(list)

Converts commma delimited string to a vector of Int64
"""
function convert_to_vector(List)
    return parse.(Int64, split(List, ","))
end

###########################
# Network Level Thickness #
###########################
function network_thickness(NetworkID, connection)
    PersistenceDiagram = compute_persistence_diagram(NetworkID, connection)

    H0Epsilons = Tables.matrix(PersistenceDiagram[1])
    H1Epsilons = Tables.matrix(PersistenceDiagram[2])
    H2Epsilons = Tables.matrix(PersistenceDiagram[3])

    H1Count = length(PersistenceDiagram[2])
    H1CapIdx = (.90 * H1Count) |> floor |> Int
    H1CapEpsilon = sort(H1Epsilons[:,2],)[H1CapIdx]

    H2Count = length(PersistenceDiagram[3])
    H2CapIdx = (.90 * H2Count) |> floor |> Int
    H2CapEpsilon = sort(H2Epsilons[:,2],)[H2CapIdx]

    EpsilonCap = max(H2CapEpsilon, H1CapEpsilon)

    SignificantEpsilons::Vector{Float64} = vcat(H0Epsilons[:,1], H0Epsilons[:, 2], H1Epsilons[:,1], H1Epsilons[:, 2], H2Epsilons[:,1], H2Epsilons[:, 2])
    SignificantEpsilons = SignificantEpsilons |> unique |> sort

    Ranks = DataFrame(
        dist = SignificantEpsilons, 
        h0 = zeros(length(SignificantEpsilons)), 
        h1 = zeros(length(SignificantEpsilons)), 
        h2 = zeros(length(SignificantEpsilons)), 
        sum = zeros(length(SignificantEpsilons))
    )

    # Iterate over h0s
    for row in eachrow(H0Epsilons)
        Ranks[Ranks.dist .>= row[1] .&& Ranks.dist .< row[2], [2,5]] .+= 1.0
    end

    # Iterate over h1s
    for row in eachrow(H1Epsilons)
        Ranks[Ranks.dist .>= row[1] .&& Ranks.dist .< row[2], [3,5]] .+= 1.0
    end

    # Iterate over h2s
    for row in eachrow(H2Epsilons)
        Ranks[Ranks.dist .>= row[1] .&& Ranks.dist .< row[2], [4,5]] .+= 1.0
    end
    
    Ranks = Ranks[Ranks.dist .< EpsilonCap, :]
    RanksComputed = select(Ranks, :dist, :h0, :h1, :h2, :sum, [:h1, :h2, :sum] => ((h1, h2, sum) -> (h2 .- h1)./sum) => :tau)

    return RanksComputed
end

function network_persistence_diagram(NetworkID, connection)
    SCM = network_SCM_to_matrix(386, NetworkID, connection)
    SCM = 1.0 ./ SCM
    DistanceMatrix = floyd_warshall_shortest_paths(Graph(SCM), SCM).dists
    PersistenceDiagram = ripserer(DistanceMatrix; dim_max=2)
    
    return PersistenceDiagram
end

function network_scm_to_matrix(PopulationSize, NetworkID, connection)
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

################
# New versions #
################

function thickness_metrics_2(connection; SaveResults = false, Population = 386)
    OutbreakThreshold = 0.2*Population
    FailedSeedingThreshold = 0.00*Population

    # Compute mean and variance of thickness for each mask/vask level agg. by network level
    query = """
    WITH ThicknessTable AS (
        SELECT 
            NetworkDim.NetworkID,
            BehaviorDim.BehaviorID,
            PersistenceLoad.EpidemicID,
            EpidemicDim.InfectedTotal,
            EpidemicDim.InfectedMax,
            EpidemicDim.PeakDay,
            BehaviorDim.MaskVaxID,
            PersistenceLoad.Distance,
            PersistenceLoad.H0Count,
            PersistenceLoad.H1Count,
            PersistenceLoad.H2Count,
            (CAST(H2Count AS DECIMAL) - CAST(H1Count AS DECIMAL))/(CAST(H1Count AS DECIMAL) + CAST(H2Count AS Decimal)) AS Thickness
        FROM PersistenceLoad
        JOIN EpidemicDim ON EpidemicDim.EpidemicID = PersistenceLoad.EpidemicID
        JOIN BehaviorDim ON BehaviorDim.BehaviorID = EpidemicDim.BehaviorID
        JOIN NetworkDim ON NetworkDim.NetworkID = BehaviorDim.NetworkID
    )
    SELECT
        ThicknessTable.NetworkID,
        ThicknessTable.BehaviorID,
        ThicknessTable.EpidemicID,
        ThicknessTable.MaskVaxID,
        ThicknessTable.InfectedTotal,
        ThicknessTable.InfectedMax,
        ThicknessTable.PeakDay,
        CASE WHEN ThicknessTable.InfectedTotal <= $FailedSeedingThreshold THEN -1 WHEN ThicknessTable.InfectedTotal <= $OutbreakThreshold THEN 0 ELSE 1 END AS Outbreak,
        ThicknessTable.Distance,
        ThicknessTable.Thickness
    FROM ThicknessTable
    GROUP BY 
        ThicknessTable.NetworkID, 
        ThicknessTable.BehaviorID, 
        ThicknessTable.EpidemicID,
        ThicknessTable.InfectedTotal, 
        ThicknessTable.MaskVaxID, 
        ThicknessTable.InfectedMax, 
        ThicknessTable.PeakDay,
        ThicknessTable.Distance,
        ThicknessTable.Thickness
    ORDER BY
        ThicknessTable.NetworkID,
        ThicknessTable.BehaviorID, 
        ThicknessTable.EpidemicID,
        ThicknessTable.MaskVaxID, 
        ThicknessTable.Distance
    """
    ResultDF = run_query(query, connection) |> DataFrame    
    SaveResults && CSV.write("thickness_metrics.csv", ResultDF)
    return ResultDF
    
end

function mean_thickness_bulk(data)
    ResultDF = DataFrame(
        EpidemicID = Int64[],
        Outbreak = Int64[],
        MeanThickness = Float64[],
        VarianceThickness = Float64[],
        MaskVaxID = Int64[]
    )
    EpidemicIDs = convert.(Int64, unique(data, :EpidemicID)[:, :EpidemicID])
    for EpidemicID in EpidemicIDs
        target_data = data[data.EpidemicID .== EpidemicID, :]
        computed_mean = mean_thickness(target_data)
        computed_variance = variance_thickness(target_data, computed_mean)
        append!(ResultDF, 
            DataFrame(
                EpidemicID = EpidemicID,
                Outbreak = data[data.EpidemicID .== EpidemicID, :Outbreak][1],
                MeanThickness = computed_mean,
                VarianceThickness = computed_variance,
                MaskVaxID = data[data.EpidemicID .== EpidemicID, :MaskVaxID][1]
            )
        )
    end
    return ResultDF
end

"""
    average_thickness(data)

Computes mean thickness using the reimann sum
"""
function mean_thickness(data)
    running_sum = 0
    for (i, row) in enumerate(eachrow(data))
        # Break condition
        i == nrow(data) && break

        running_sum += row.Thickness * (data[i+1, :Distance] - data[i, :Distance])
    end

    return running_sum / (data[nrow(data), :Distance] - data[1, :Distance])

end

function variance_thickness(data, mean)
    running_sum = 0
    for row in eachrow(data)
        running_sum += (row.Thickness - mean)^2
    end
    return running_sum/nrow(data)
end


# Build a query to get epidemicid and NetworkID
query = """
SELECT EpidemicID, NetworkID
FROM EpidemicDim
WHERE EpidemicID IN (
    SELECT EpidemicID
    FROM EpidemicDim
    WHERE BehaviorID IN (
        SELECT BehaviorID
        FROM BehaviorDim
        WHERE MaskVaxID IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    )
)
"""