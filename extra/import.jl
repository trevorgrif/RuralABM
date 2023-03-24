using DuckDB, DataFrames, Parquet, CSV

function load_allsmall(con)
    load_all("Random", "Random", 30, "data\\live_data\\small_RR", con)
    load_all("Watts", "Watts", 30, "data\\live_data\\small_WW", con)
    load_all("Watts", "Random", 30, "data\\live_data\\small_WR", con)
    load_all("Random", "Watts", 30, "data\\live_data\\small_RW", con)
end

function run_query(query, connection)
    DBInterface.execute(connection, query)
end

function create_con()
    return DBInterface.connect(DuckDB.DB, "data/GDWLND.duckdb")
end

######################################
#   Table & Key Sequence Creation    #
######################################

function create_tables(connection)
    query_list = []
    # append!(query_list, ["CREATE OR REPLACE TABLE PopulationDim (PopulationID USMALLINT PRIMARY KEY, Description VARCHAR)"])
    # append!(query_list, ["CREATE OR REPLACE TABLE PopulationLoad (PopulationID USMALLINT, AgentID INT, HouseID INT, AgeRangeID VARCHAR, Sex VARCHAR, IncomeRange VARCHAR, PRIMARY KEY (PopulationID, AgentID))"])
    append!(query_list, ["CREATE OR REPLACE TABLE TownDim (TownID USMALLINT PRIMARY KEY, PopulationID USMALLINT, BusinessCount INT, HouseCount INT, SchoolCount INT, DaycareCount INT, GatheringCount INT, AdultCount INT, ElderCount INT, ChildCount INT, EmptyBusinessCount INT, MaskDistributionType VARCHAR , VaxDistributionType VARCHAR)"])
    append!(query_list, ["CREATE OR REPLACE TABLE BusinessTypeDim (BusinessTypeID USMALLINT PRIMARY KEY, Description VARCHAR)"])
    append!(query_list, ["CREATE OR REPLACE TABLE BusinessLoad (TownID USMALLINT, BusinessID INT, BusinessTypeID INT, EmployeeCount INT, PRIMARY KEY (TownID, BusinessID))"])
    append!(query_list, ["CREATE OR REPLACE TABLE HouseholdLoad (TownID USMALLINT, HouseholdID INT, ChildCount INT, AdultCount INT, ElderCount INT, PRIMARY KEY (TownID, HouseholdID))"])
    append!(query_list, ["CREATE OR REPLACE TABLE NetworkDim (NetworkID USMALLINT PRIMARY KEY, TownID INT, ConstructionLengthDays INT)"])
    append!(query_list, ["CREATE OR REPLACE TABLE NetworkSCMLoad (NetworkID USMALLINT, Agent1 INT, Agent2 INT, Weight INT, PRIMARY KEY (NetworkID, Agent1, Agent2))"])
    append!(query_list, ["CREATE OR REPLACE TABLE BehaviorDim (BehaviorID USMALLINT PRIMARY KEY, NetworkID INT, MaskVaxID INT)"])
    append!(query_list, ["CREATE OR REPLACE TABLE MaskVaxDim (MaskVaxID USMALLINT PRIMARY KEY, MaskPortion UTINYINT, VaxPortion UTINYINT)"])
    append!(query_list, ["CREATE OR REPLACE TABLE AgentLoad (BehaviorID UINTEGER, AgentID INT, AgentHouseholdID INT, IsMasking INT, IsVaxed INT, PRIMARY KEY (BehaviorID, AgentID))"])
    append!(query_list, ["CREATE OR REPLACE TABLE EpidemicDim (EpidemicID UINTEGER PRIMARY KEY, BehaviorID UINTEGER, InfectedTotal USMALLINT, InfectedMax USMALLINT, PeakDay USMALLINT, RecoveredTotal USMALLINT, RecoveredMasked USMALLINT, RecoveredVaccinated USMALLINT, RecoveredMaskAndVax USMALLINT)"])
    append!(query_list, ["CREATE OR REPLACE TABLE TransmissionLoad (EpidemicID UINTEGER, AgentID USMALLINT, InfectedBy USMALLINT, InfectionTimeHour UINTEGER)"])
    append!(query_list, ["CREATE OR REPLACE TABLE EpidemicLoad (EpidemicID UINTEGER, Day USMALLINT, Symptomatic USMALLINT, Recovered USMALLINT, PopulationLiving USMALLINT, PRIMARY KEY (EpidemicID, Day))"])

    for query in query_list
        run_query(query, connection)
    end
end

function drop_tables(connection)
    query_list = []
    append!(query_list, ["DROP TABLE IF EXISTS TownDim"])
    append!(query_list, ["DROP TABLE IF EXISTS BusinessTypeDim"])
    append!(query_list, ["DROP TABLE IF EXISTS BusinessLoad"])
    append!(query_list, ["DROP TABLE IF EXISTS HouseholdLoad"])
    append!(query_list, ["DROP TABLE IF EXISTS NetworkDim"])
    append!(query_list, ["DROP TABLE IF EXISTS NetworkSCMLoad"])
    append!(query_list, ["DROP TABLE IF EXISTS BehaviorDim"])
    append!(query_list, ["DROP TABLE IF EXISTS MaskVaxDim"])
    append!(query_list, ["DROP TABLE IF EXISTS AgentLoad"])
    append!(query_list, ["DROP TABLE IF EXISTS EpidemicDim"])
    append!(query_list, ["DROP TABLE IF EXISTS TransmissionLoad"])
    append!(query_list, ["DROP TABLE IF EXISTS EpidemicLoad"])
    append!(query_list, ["DROP VIEW IF EXISTS EpidemicSCMLoad_1"])
    append!(query_list, ["DROP VIEW IF EXISTS EpidemicSCMLoad_2"])
    append!(query_list, ["DROP VIEW IF EXISTS EpidemicSCMLoad_3"])
    append!(query_list, ["DROP VIEW IF EXISTS EpidemicSCMLoad_4"])

    for query in query_list
        run_query(query, connection)
    end
end

function create_sequences(connection)
    query_list = []
    append!(query_list, ["CREATE SEQUENCE TownDimSequence START 1"])
    append!(query_list, ["CREATE SEQUENCE BusinessTypeDimSequence START 1"])
    append!(query_list, ["CREATE SEQUENCE NetworkDimSequence START 1"])
    append!(query_list, ["CREATE SEQUENCE BehaviorDimSequence START 1"])
    append!(query_list, ["CREATE SEQUENCE MaskVaxDimSequence START 1"])
    append!(query_list, ["CREATE SEQUENCE EpidemicDimSequence START 1"])

    for query in query_list
        run_query(query, connection)
    end
end

function drop_sequences(connection)
    query_list = []
    append!(query_list, ["DROP SEQUENCE IF EXISTS TownDimSequence"])
    append!(query_list, ["DROP SEQUENCE IF EXISTS BusinessTypeDimSequence"])
    append!(query_list, ["DROP SEQUENCE IF EXISTS NetworkDimSequence"])
    append!(query_list, ["DROP SEQUENCE IF EXISTS BehaviorDimSequence"])
    append!(query_list, ["DROP SEQUENCE IF EXISTS MaskVaxDimSequence"])
    append!(query_list, ["DROP SEQUENCE IF EXISTS EpidemicDimSequence"])

    for query in query_list
        run_query(query, connection)
    end
end

######################################
#         High Level Loading         #
######################################
function load_all(MaskDistributionType, VaxDistributionType, ConstructionLengthDays, directory, connection)
    print("Loading Town Level\n") 
    TownID = load_TownLevel(MaskDistributionType, VaxDistributionType, directory, connection)
    print("Loading Network Level\n")
    NetworkIDs = load_NetworkLevel(TownID, ConstructionLengthDays, directory, connection)
    print("Loading Behavior Level\n")
    BehaviorIDs = load_BehaviorLevel(NetworkIDs, directory, connection)
    print("Loading Epidemic Level\n")
    load_EpidemicLevel(TownID, BehaviorIDs, directory, connection)
end

function create_all(connection)
    create_tables(connection)
    create_sequences(connection)
end

function drop_all(connection)
    drop_tables(connection)
    drop_sequences(connection)
end

######################################
#         Town Level Loading         #
######################################
function load_TownLevel(MaskDistributionType, VaxDistributionType, directory, connection)
    TownID = load_TownDim(MaskDistributionType, VaxDistributionType, directory, connection)
    load_BusinessLoad(TownID, directory, connection)
    load_HouseholdLoad(TownID, directory, connection)
    #load_BusinessTypeDim(connection) # Static Dim

    return TownID
end

function load_TownDim(MaskDistributionType, VaxDistributionType, directory, connection)
    # Set file location
    file = "town_summary.csv"

    # Load raw csv into temp table
    query = "CREATE TEMPORARY TABLE temp.TownDimTemp AS SELECT * FROM read_csv_auto('$(directory)\\$file')"
    run_query(query, connection)

    # Port from temp table to table
    query = """
        INSERT INTO main.TownDim 
        SELECT nextval('TownDimSequence') AS TownID, 1, *, '$MaskDistributionType', '$VaxDistributionType' FROM temp.TownDimTemp
        RETURNING TownID
    """
    TownID = DataFrame(run_query(query, connection))[1,1]

    # Clean Up
    query = "DROP TABLE temp.TownDimTemp"
    run_query(query, connection)

    return TownID
end

function load_BusinessLoad(TownID, directory, connection)
    # Set file location
    file = "town_businesses.csv"
    
    # Load raw csv into temp table
    query = "CREATE TEMPORARY TABLE temp.BusinessLoadTemp AS SELECT ROW_NUMBER() OVER (ORDER BY (SELECT 1)) AS BusinessID, * FROM read_csv_auto('$(directory)\\$file')"
    run_query(query, connection)
    
    # Port from temp table to table
    run_query("INSERT INTO main.BusinessLoad SELECT $TownID, * FROM temp.BusinessLoadTemp", connection)
    
    # Clean Up
    query = "DROP TABLE temp.BusinessLoadTemp"
    run_query(query, connection)
end

function load_HouseholdLoad(TownID, directory, connection)
    # Set file location
    file = "town_households.csv"
    
    # Load raw csv into temp table
    query = "CREATE TEMPORARY TABLE temp.HouseholdLoadTemp AS SELECT * FROM read_csv_auto('$(directory)\\$file')"
    run_query(query, connection)
    
    # Port from temp table to table
    query = "INSERT INTO main.HouseholdLoad SELECT $TownID, ID, childrenCount, adultCount, retireeCount FROM temp.HouseholdLoadTemp"
    run_query(query, connection)
    
    # Clean Up
    query = "DROP TABLE temp.HouseholdLoadTemp"
    run_query(query, connection)
end

function load_BusinessTypeDim(connection)
    query_list = []
    append!(query_list, "INSERT INTO BusinessTypeDim VALUES (0, Private Business)")
    append!(query_list, "INSERT INTO BusinessTypeDim VALUES (1, Public Facing Business)")
    
    for query in query_list
        run_query(query, connection)
    end
end

######################################
#        Network Level Loading       #
######################################

function load_NetworkLevel(TownID, ConstructionLengthDays, directory, connection)
    PrecontagionSCMs = CSV.read("$(directory)\\SCM\\precontagion.csv", DataFrame, limit = 1) 
    NetworkCount = ncol(PrecontagionSCMs)
    NetworkIDs = load_NetworkDim(TownID, ConstructionLengthDays, NetworkCount, connection)
    load_NetworkSCMLoad(NetworkIDs, directory, connection)
    return NetworkIDs
end

function load_NetworkDim(TownID, ConstructionLengthDays, NetworkCount, connection)
    NetworkIDs::Vector{Int64} = []
    NetworkDimAppender = DuckDB.Appender(connection, "NetworkDim")
    for i in 1:NetworkCount
        query = """
        SELECT nextval('NetworkDimSequence')
        """
        NetworkID = DataFrame(run_query(query, connection))[1,1]
        append!(NetworkIDs, [NetworkID])

        DuckDB.append(NetworkDimAppender, NetworkID)
        DuckDB.append(NetworkDimAppender, TownID)
        DuckDB.append(NetworkDimAppender, ConstructionLengthDays)
        DuckDB.end_row(NetworkDimAppender)
    end
    DuckDB.close(NetworkDimAppender)
    return NetworkIDs
end

function load_NetworkSCMLoad(NetworkIDs, directory, connection)
    # Load Data
    PrecontagionSCMs = CSV.read("$(directory)\\SCM\\precontagion.csv", DataFrame) 
    NetworkSCMAppender = DuckDB.Appender(connection, "NetworkSCMLoad")
    NetworkIDItr = 1
    
    for SCM in eachcol(PrecontagionSCMs)
        population = first(SCM)
        NetworkID = NetworkIDs[NetworkIDItr]
        NetworkSCMItr = 2        
        for agent1 in 1:population
            for agent2 in (agent1+1):population
                DuckDB.append(NetworkSCMAppender, NetworkID)
                DuckDB.append(NetworkSCMAppender, agent1)
                DuckDB.append(NetworkSCMAppender, agent2)
                DuckDB.append(NetworkSCMAppender, SCM[NetworkSCMItr])
                DuckDB.end_row(NetworkSCMAppender)

                NetworkSCMItr = NetworkSCMItr + 1
            end
        end
        NetworkIDItr = NetworkIDItr + 1
    end
    DuckDB.close(NetworkSCMAppender)
end

######################################
#        Behavior Level Loading      #
######################################

function load_BehaviorLevel(NetworkIDs, directory, connection)
    load_MaskVaxDim(directory, connection)
    BehaviorIDs = load_BehaviorDim(NetworkIDs, directory, connection)
    load_AgentLoad(BehaviorIDs, directory, connection)
    return BehaviorIDs
end

function load_BehaviorDim(NetworkIDs, directory, connection)
    BehaviorIDs::Vector{Int64} = []
    BehaviorDimAppender = DuckDB.Appender(connection, "BehaviorDim")
    for NetworkID in NetworkIDs
        NetworkIDX = findfirst(x -> x == NetworkID, NetworkIDs)
        files = filter(t -> occursin("Agent",t), readdir("$directory\\TN\\$(lpad(NetworkIDX, 3, "0"))"))
        for file in files
            MaskPortion = file[1:2]
            VaxPortion = file[4:5]
            query = """
            SELECT MaskVaxID, nextval('BehaviorDimSequence')
            FROM MaskVaxDim
            WHERE MaskPortion = $MaskPortion
            AND VaxPortion = $VaxPortion
            """
            ResultDF = DataFrame(run_query(query, connection))
            MaskVaxID = ResultDF[1,1]
            BehaviorID = ResultDF[1,2]

            DuckDB.append(BehaviorDimAppender, BehaviorID)
            DuckDB.append(BehaviorDimAppender, NetworkID)
            DuckDB.append(BehaviorDimAppender, MaskVaxID)
            DuckDB.end_row(BehaviorDimAppender)

            append!(BehaviorIDs, [BehaviorID])
        end
    end
    DuckDB.close(BehaviorDimAppender)

    return BehaviorIDs
end

function load_MaskVaxDim(directory, connection)
    SummaryDF = CSV.read("$directory\\ED\\summary_001.csv", DataFrame)
    DuckDB.register_data_frame(connection, SummaryDF, "MaskVaxTemp")
    MaskVaxLevelDF = DataFrame(run_query("SELECT DISTINCT MaskLvl, VaxLvl FROM MaskVaxTemp", connection))

    for row in eachrow(MaskVaxLevelDF)
        query = """
        SELECT COUNT(1) FROM MaskVaxDim
        WHERE MaskPortion = $(row[1])
        AND VaxPortion = $(row[2]) 
        """
        IDExists = DataFrame(run_query(query, connection))[1,1]

        if IDExists == true
            continue
        end

        query = """
        INSERT INTO MaskVaxDim 
        VALUES (nextval('MaskVaxDimSequence'), $(row[1]), $(row[2]))
        """
        run_query(query, connection)
    end

    run_query("DROP VIEW IF EXISTS MaskVaxTemp", connection)

end

function load_AgentLoad(BehaviorIDs, directory, connection)
    NetworkIDXs = readdir("$directory\\TN")
    AgentLoadAppender = DuckDB.Appender(connection, "AgentLoad")
    BehaviorIDItr = 1
    for NetworkIDX in NetworkIDXs
        files = filter(t -> occursin("Agent",t), readdir("$directory\\TN\\$NetworkIDX"))
        for file in files
            AgentLoadCSV = CSV.File("$directory\\TN\\$NetworkIDX\\$file")
           
            for row in AgentLoadCSV
                DuckDB.append(AgentLoadAppender, BehaviorIDs[BehaviorIDItr])
                DuckDB.append(AgentLoadAppender, row[1])
                DuckDB.append(AgentLoadAppender, row[2])
                DuckDB.append(AgentLoadAppender, Int(row[3]))
                DuckDB.append(AgentLoadAppender, Int(row[4]))
                DuckDB.end_row(AgentLoadAppender)
            end
            BehaviorIDItr = BehaviorIDItr + 1
        end
    end
    DuckDB.close(AgentLoadAppender)
    return
end

######################################
#        Epidemic Level Loading      #
######################################

function load_EpidemicLevel(TownID, BehaviorIDs, directory, connection)
    EpidemicIDs = load_EpidemicDim(BehaviorIDs, directory, connection)
    @show EpidemicIDs
    print("Converting SCM to PARQUET\n")
    load_EpidemicSCMtoParquet(TownID, EpidemicIDs, directory)
    load_EpidemicSCMLoad(TownID, directory, connection)
    load_EpidemicLoad(EpidemicIDs, directory, connection)
    load_TransmissionLoad(EpidemicIDs, directory, connection)
end

function load_EpidemicDim(BehaviorIDs, directory, connection)
    EpidemicIDs::Vector{Int64} = []
    summary_files = filter(t -> occursin(".csv",t), readdir("$directory\\ED"))
    EpidemicDimAppender = DuckDB.Appender(connection, "EpidemicDim")

    BehaviorIDsItr = 0
    for summary_file in summary_files
        SummaryDF = CSV.File("$directory\\ED\\$summary_file")
        RowCount = 1
        for row in SummaryDF

            if (RowCount % 100) == 1
                BehaviorIDsItr +=1
            end

            query = """
            SELECT nextval('EpidemicDimSequence')
            """
            EpidemicID = DataFrame(run_query(query, connection))[1,1]
            append!(EpidemicIDs, [EpidemicID])
            DuckDB.append(EpidemicDimAppender, EpidemicID)
            DuckDB.append(EpidemicDimAppender, BehaviorIDs[BehaviorIDsItr])
            DuckDB.append(EpidemicDimAppender, row[5])
            DuckDB.append(EpidemicDimAppender, row[6])
            DuckDB.append(EpidemicDimAppender, row[7])
            DuckDB.append(EpidemicDimAppender, row[8])
            DuckDB.append(EpidemicDimAppender, row[9])
            DuckDB.append(EpidemicDimAppender, row[10])
            DuckDB.append(EpidemicDimAppender, row[11])
            DuckDB.end_row(EpidemicDimAppender)
            
            RowCount += 1
        end
    end

    DuckDB.close(EpidemicDimAppender)
    return EpidemicIDs
end

function load_EpidemicLoad(EpidemicIDs, directory, connection)
    EpidemicLoadDirs = filter(t -> !occursin(".csv",t), readdir("$directory\\ED"))
    EpidemicIDItr = 1
    EpidemicLoadAppender = DuckDB.Appender(connection, "EpidemicLoad")
    for EpidemicLoadDir in EpidemicLoadDirs
        EpidemicFiles = readdir("$directory\\ED\\$EpidemicLoadDir")
        for EpidemicFile in EpidemicFiles
            EpidemicID = EpidemicIDs[EpidemicIDItr]

            EpidemicCSV = CSV.File("$directory\\ED\\$EpidemicLoadDir\\$EpidemicFile")

            for row in EpidemicCSV
                DuckDB.append(EpidemicLoadAppender, EpidemicID)
                DuckDB.append(EpidemicLoadAppender, row[1])
                DuckDB.append(EpidemicLoadAppender, row[2])
                DuckDB.append(EpidemicLoadAppender, row[3])
                DuckDB.append(EpidemicLoadAppender, row[4])
                DuckDB.end_row(EpidemicLoadAppender)
            end
            EpidemicIDItr = EpidemicIDItr + 1
        end
    end
    DuckDB.close(EpidemicLoadAppender)
end

function load_TransmissionLoad(EpidemicIDs, directory, connection)
    TransmissionLoadDirs = filter(t -> !occursin(".csv",t), readdir("$directory\\TN"))
    EpidemicIDItr = 1
    TransmissionLoadAppender = DuckDB.Appender(connection, "TransmissionLoad")
    for TransmissionLoadDir in TransmissionLoadDirs
        TransmissionFiles = filter(t -> !occursin("Agent",t), readdir("$directory\\TN\\$TransmissionLoadDir"))
        for TransmissionFile in TransmissionFiles
            EpidemicID = EpidemicIDs[EpidemicIDItr]
            TransmissionCSV = CSV.File("$directory\\TN\\$TransmissionLoadDir\\$TransmissionFile")

            for row in TransmissionCSV
                DuckDB.append(TransmissionLoadAppender, EpidemicID)
                DuckDB.append(TransmissionLoadAppender, row[1])
                DuckDB.append(TransmissionLoadAppender, row[2])
                DuckDB.append(TransmissionLoadAppender, row[3])
                DuckDB.end_row(TransmissionLoadAppender)
            end
            EpidemicIDItr = EpidemicIDItr + 1
        end
    end
    DuckDB.close(TransmissionLoadAppender)
end

function load_EpidemicSCMLoad(TownID, directory, connection)
    query = """
    CREATE VIEW EpidemicSCMLoad_$TownID AS SELECT * FROM '$directory\\..\\PARQUET\\$(TownID)_*.parquet';
    """
    run_query(query, connection)
end

function old_load_EpidemicSCMLoad(EpidemicIDs, directory, connection)
    EpidemicSCMFiles = filter(t -> occursin("post",t), readdir("$directory\\SCM"))
    EpidemicSCMAppender = DuckDB.Appender(connection, "EpidemicSCMLoad")
    
    EpidemicIDsStartIDX = 0
    for file in EpidemicSCMFiles
        EpidemicSCMs = CSV.File("$(directory)\\SCM\\$file")
        Population = EpidemicSCMs[1][1]
        KeyPoints = get_ArithmeticKeyPoints(Population-1)

        WeightItr = 1
        KeyPointsItr = 1
        for WeightRow in EpidemicSCMs[2:end] # Skip the population size row
            # Compute Agent1 and Agent2
            if (WeightItr) > KeyPoints[KeyPointsItr+1]
                KeyPointsItr = KeyPointsItr+1
            end
            agent1 = KeyPointsItr
            agent2 = (WeightItr) - (KeyPoints[KeyPointsItr]) + (KeyPointsItr - 1)
            
            # Append each weight 
            for i in 1:length(WeightRow)
                DuckDB.append(EpidemicSCMAppender, EpidemicIDs[EpidemicIDsStartIDX + i])
                DuckDB.append(EpidemicSCMAppender, agent1)
                DuckDB.append(EpidemicSCMAppender, agent2)
                DuckDB.append(EpidemicSCMAppender, WeightRow[i])
                DuckDB.end_row(EpidemicSCMAppender)
            end
            WeightItr = WeightItr + 1
        end
        EpidemicIDsStartIDX = EpidemicIDsStartIDX + length(EpidemicSCMs[1])
    end
    DuckDB.close(EpidemicSCMAppender)
end

function get_ArithmeticKeyPoints(n)
    KeyPoints = []
    for i in 1:n
        append!(KeyPoints, get_S(n,i))
    end
    return KeyPoints
end

function get_S(n,i)
    return Int((i-1)*(n+1-(i/2.0)))
end

function load_EpidemicSCMtoParquet(TownID, EpidemicIDs, directory)
    EpidemicSCMFiles = filter(t -> occursin("post",t), readdir("$directory\\SCM"))
    EpidemicIDsItr = 1
    for file in EpidemicSCMFiles
        FileDF = DataFrame(CSV.File("$directory\\SCM\\$file"))

        NewDF = DataFrame(EpidemicID = Int[], SCM =String[])
        for col in eachcol(FileDF)
            append!(NewDF, DataFrame(EpidemicID = EpidemicIDs[EpidemicIDsItr], SCM = join(col[2:end], ",")))
            EpidemicIDsItr += 1
        end

        write_parquet("$directory\\..\\PARQUET\\$(TownID)_$(file[1:23]).parquet", NewDF)
    end
end

function import_population_data(connection)
    population = DataFrame(CSV.File("data\\example_towns\\large_town\\population.csv"))
    select!(population, Not([:Ind]))
    rename!(population, :Column1 => :AgentID)
    rename!(population, :house => :HouseID)
    rename!(population, :age => :AgeRange)
    rename!(population, :sex => :Sex)
    rename!(population, :income => :IncomeRange)
    population[!,:PopulationID] .= 2
    select!(population, :PopulationID, Not([:PopulationID]))

    DuckDB.register_data_frame(connection, population, "population")
    run_query("INSERT INTO PopulationLoad SELECT * FROM population WHERE HouseID <> 'NA'", connection)
    run_query("DROP VIEW population", connection)
end

function extract_first_column_csv()
    open("data\\live_data\\large_test\\postcontagion_80_00_001.csv") do f1
        open("data\\live_data\\large_test\\large_view.csv", "w") do f2
            for l in eachline(f1)
                v1 = parse(Int32, l[1:findfirst(==(','), l)-1])
                write(f2, "$v1\n")
            end
        end
    end
end