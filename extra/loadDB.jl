using DuckDB

#================#
# Helper Methods #
#================#

function run_query(query, connection)
    DBInterface.execute(connection, query)
end

function create_con()
    return DBInterface.connect(DuckDB.DB, "data/GDWLND.duckdb")
end

#================#
# Main Functions #
#================#
"""
    load_db_exported(filepath)

Loads db_exported into `data\GDWLND.duckdb`.
"""
function load_db_exported(filepath)
    # Check if machine is windows or linux
    if Sys.iswindows()
        # Check `data` directory exists
        if !isdir("data")
            mkdir("data")
        end

        # Create connection variable
        con = create_con()

        # check filepath\schema.sql exists
        if !isfile("$(filepath)\\schema.sql")
            error("$(filepath)\\schema.sql does not exist")
        end

        # read filepath\schema.sql line by line ignoring empty lines
        open("$(filepath)\\schema.sql") do file
            for line in eachline(file)
                if line != "" && line != ";"
                    # run query
                    run_query("""$line""", con)
                end
            end
        end

        # check filepath\load.sql exists
        if !isfile("$(filepath)\\load.sql")
            error("$(filepath)\\load.sql does not exist")
        end

        # read filepath\load.sql line by line ignoring empty lines
        open("$(filepath)\\load.sql") do file
            for line in eachline(file)
                if line != "" && line != ";"
                    # run query
                    run_query("""$line""", con)
                end
            end
        end
    else
        # Check `data` directory exists
        if !isdir("data")
            mkdir("data")
        end

        # Create connection variable
        con = create_con()

        # check filepath/schema.sql exists
        if !isfile("$(filepath)/schema.sql")
            error("$(filepath)/schema.sql does not exist")
        end

        # read filepath/schema.sql line by line ignoring empty lines
        open("$(filepath)/schema.sql") do file
            for line in eachline(file)
                if line != "" && line != ";"
                    # run query
                    run_query("""$line""", con)
                end
            end
        end

        # check filepath/load.sql exists
        if !isfile("$(filepath)/load.sql")
            error("$(filepath)/load.sql does not exist")
        end

        # read filepath/load.sql line by line ignoring empty lines
        open("$(filepath)/load.sql") do file
            for line in eachline(file)
                if line != "" && line != ";"
                    # run query
                    run_query("""$line""", con)
                end
            end
        end
    end
end

"""
    test_database()

Tests the database by running a query and returning the result.
"""
function test_database()
    # Create connection variable
    con = create_con()

    # Run query
    result = run_query("SELECT * FROM TownDim LIMIT 10", con)

    # Close connection
    DBInterface.close(con)

    # Return result
    return result
end
