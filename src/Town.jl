function demographics(model)
    Population = DataFrame(agent = Int64[], age = Int[], sex = Symbol[], home = Int64[] )
    for agent in allagents(model)
        push!(Population,[agent.id,agent.age,agent.sex,agent.home])
    end
    return Population
end

function get_age(agent)
    cut = split(agent.age,"-")
    return string(parse(Int,cut[1]):parse(Int,cut[2]) |> rand)
end

#============================================================
Relabels the ":Type" symbol for all vertices with matching
:SIC code as SIC_vector in the metagraph (Graph) to new_type.
Returns all identified vertices as a collection.
============================================================#
function relabel_type_from_SIC!(Graph, SIC_vector, new_type)
    # Find all matching SIC code locations
    set = collect(filter_vertices(Graph, (g,v)->get_prop(g,v,:SIC) ∈ SIC_vector))

    # Change :Type to new_type for all matching locations
    for items in set
        set_prop!(Graph, items, :Type, new_type)
    end

    return set
end

#============================================================
Assign distribution (alpha) of households to business groups
============================================================#
function link_households_to_business!(Graph, group_type; alpha = 0.5)
    # Collect groups from town structure
    houses = collect(filter_vertices(Graph, (g,v)->get_prop(g,v,:Type) == :House))
    target_group = collect(filter_vertices(Graph, (g,v)->get_prop(g,v,:Type) == group_type))

    # For some percentage of households, attach a random element from the target group
    for house in houses
        if rand() > alpha
            rand_group_ele = target_group |> rand
            set_prop!(Graph, house, :Community_Gathering, rand_group_ele)
        end
    end
end

function print_type(Graph, type)
    print("Printing Type\n")
    set = collect(filter_vertices(Graph, (g,v)->get_prop(g,v,:Type) == type))
    for place in set
        print(place)
        print("\n")
    end
end

#============================================================
Build town structure based on CSV file
============================================================#
function build_town(household_file_path::String, business_file_path::String;
                    βrange=(0.75,1.1))
    if VERBOSE
        print("Building Town...\n")
    end

    # Import household data into Town object
    Town = CSV.File(household_file_path) |> DataFrame

    # Cleaning household data
    filter!(x -> x.house != "NA", Town) #Remove houseless individuals
    Town.house = parse.(Int,Town.house) #Convert house from string to int

    # Fix age groups of households (probably a better way to do this?)
    replace!(Town.age,
            "85-NA" => "85-100")
    for row in eachrow(Town)
        row.age = get_age(row)
    end
    Town.age = parse.(Int,Town.age)

    # Import business structure into Businesses object
    Businesses = CSV.File(business_file_path) |> DataFrame

    # Model Properties & Space based on Workplaces, Schools, and Homes
    njobs = Businesses.EMPNUM |> sum |> floor |> Int64

    nhouses = max(Town.house...)

    #Adding Household Structure
    town_structure = MetaGraph(0)
    town_structure = add_household_structure!(town_structure, nhouses)

    # Adding business structure to town_structure
    town_structure = add_business_structure!(town_structure, Businesses)
    nbusinesses = nv(town_structure) - nhouses

    # Relabeling special businesses and storing all types in containers
    school_SIC = ["821101","821102","821103"]
    daycare_SIC = ["835100","835101","835102","835103"]
    church_SIC = ["866106","866107"]
    schools = relabel_type_from_SIC!(town_structure, school_SIC, :School)
    daycares = relabel_type_from_SIC!(town_structure, daycare_SIC, :DayCare)
    churches = relabel_type_from_SIC!(town_structure, church_SIC, :Community_Gathering)
    businesses = collect(filter_vertices(town_structure,:Type,:Business))
    houses = collect(filter_vertices(town_structure,:Type,:House))

    # Checking for necessary business types
    town_businesses = [schools, daycares, churches, businesses]
    for bldg in town_businesses
        if isempty(bldg)
            @warn "Special business type container is empty. If town construction fails, try expanding SIC codes for" bldg
        end
    end

    # Assigning community gatherings to households
    link_households_to_business!(town_structure, :Community_Gathering)

    # Preparing Agent construction
    Children = filter(x -> x.age ≤ 18, Town)
    Adults = filter(x -> 18 < x.age, Town)
    nAgents = size(Children)[1] + size(Adults)[1]
    njobs = min(njobs, size(Adults)[1])
    Retirees = Adults[(njobs+1):end,:] # Excess adults (more than jobs available) become Retirees
    Adults = Adults[1:njobs,:] # Trim off Retirees from Adults

    # Adding Graph-Space properties
    space = GraphSpace(town_structure)
    Properties = Dict(:business => businesses,
                      :daycare => daycares,
                      :school => schools,
                      :houses => 1:nhouses,
                      :community_gathering => churches,
                      :time => 0,
                      :day => 0,
                      :shifts => [(0,8);(2,10);(4,12)],
                      :disease_parameters => Build_Disease_Parameters(βrange = βrange),
                      :behavior_parameters => Build_Behavior_Parameters(),
                      :age_parameters => Build_Age_Parameters(),
                      :risk_parameters => Build_Risk_Parameters(),
                      :TransmissionNetwork => DataFrame(agent = Int64[], infected_by = Int64[], time_infected = Int64[]),
                      :DeadAgents => DataFrame(Agent = Int64[], Home = Int64[], contact_list = SparseVector[]),
                      :Agent_Extraction_Data => DataFrame([Symbol("$(x)") for x in 1:nAgents] .=> [agent_extraction_data[] for x in 1:nAgents]),
                      :init_pop_size => nAgents,
                      :model_steps => 0
                      )

    # Intiating model construction
    if !isempty(Retirees)
        model = ABM(Union{Child,Adult,Retiree}, space; properties=Properties, warn=false)
    else
        model = ABM(Union{Child,Adult},space; properties = Properties, warn=false)
    end

    # Create Agents from Town data
    id = 0
    for child in eachrow(Children)
        id +=1
        age = child.age
        sex = Symbol(child.sex)
        house = child.house
        community_gathering = get_prop(town_structure, house, :Community_Gathering)
        if age < 5 && !isempty(daycares)
            school = rand(daycares)
        else
            school = rand(schools)
        end
        β = rand(Beta(2,3))
        global_mask_threshold = rand(Uniform(βrange...))
        local_mask_threshold = rand(Uniform(βrange...))
        agent = Child(id, house, age, sex, house, community_gathering, school, :S, 0.0, β, zeros(size(Town)[1]), false, zeros(3), false, global_mask_threshold, local_mask_threshold, 0)
        add_agent_pos!(agent,model)
    end
    for adult in eachrow(Adults)
        id += 1
        age=adult.age
        sex = Symbol(adult.sex)
        house = adult.house
        work = collect(filter_vertices(model.space.graph, jobs_open)) |> rand
        set_prop!(model.space.graph, work, :Employees, get_prop(model.space.graph, work,:Employees)+1)
        community_gathering = get_prop(town_structure, house, :Community_Gathering)
        income = 10000 #Placeholder for now, unused
        shift = rand(model.shifts)
        β = rand(Beta(2,3))
        global_mask_threshold = rand(Uniform(βrange...))
        local_mask_threshold = rand(Uniform(βrange...))
        agent = Adult(id, house, age, sex, house, work, community_gathering, income, shift, :S, 0.0, β, zeros(size(Town)[1]), false, zeros(3), false, global_mask_threshold, local_mask_threshold, 0)
        add_agent_pos!(agent,model)
    end
    for geezer in eachrow(Retirees)
        id+= 1
        age = geezer.age
        sex = Symbol(geezer.sex)
        house = geezer.house
        community_gathering = get_prop(town_structure, house, :Community_Gathering)
        income = 10000
        β = rand(Beta(2,3))
        global_mask_threshold = rand(Uniform(βrange...))
        local_mask_threshold = rand(Uniform(βrange...))
        agent = Retiree(id, house, age, sex, house, community_gathering, income, :S, 0.0, β, zeros(size(Town)[1]), false, zeros(3), false, global_mask_threshold, local_mask_threshold, 0)
        add_agent_pos!(agent,model)
    end

    # Collected Town Sturcture Data
    businessStructureDF = DataFrame()
    NumEmptyBusinesses = 0
    for shop in businesses
        numEmployed = get_prop(model.space.graph, shop, :Employees)
        bussType = get_prop(model.space.graph, shop, :business_type)
        if(numEmployed == 0)
            NumEmptyBusinesses += 0
        end
        append!(businessStructureDF, DataFrame(type = bussType, EmployeeCount = numEmployed))
    end

    houseStructureDF = DataFrame(ID = Int64[], workplaceCount = Int64[], schoolsCount = Int64[], childrenCount = Int64[], adultCount = Int64[], retireeCount = Int64[])
    for house in houses
        push!(houseStructureDF, [house, 0, 0, 0, 0, 0])
    end

    for agent in allagents(model)
        if agent isa Child
            houseStructureDF[in([agent.home]).(houseStructureDF.ID), 3] .+= 1
            houseStructureDF[in([agent.home]).(houseStructureDF.ID), 4] .+= 1
        elseif agent isa Adult
            houseStructureDF[in([agent.home]).(houseStructureDF.ID), 2] .+= 1
            houseStructureDF[in([agent.home]).(houseStructureDF.ID), 5] .+= 1
        elseif agent isa Retiree
            houseStructureDF[in([agent.home]).(houseStructureDF.ID), 6] .+= 1
        end
    end

    townDataSummaryDF = DataFrame(NumBusinesses = nbusinesses,
                                  NumHouses = nhouses,
                                  NumSchools = length(schools),
                                  NumDaycares = length(daycares),
                                  NumCommGathers = length(churches),
                                  NumAdults = size(Adults)[1],
                                  NumRetiree = size(Retirees)[1],
                                  NumChildren = size(Children)[1],
                                  NumEmptyBusinesses = NumEmptyBusinesses
                                  )
                                  
    return model, townDataSummaryDF, businessStructureDF, houseStructureDF
end

function import_contacts!(model,AdjacencyMatrix)
    for agent in allagents(model)
        agent.contact_list = AdjacencyMatrix[:,agent.id]
    end
end

@inline function rownumber(DF)
    parentindices(DF)[1]
end

#============================================================
Adds nhouses to metagraph (graph) where each house has the
following property:
    Type: House (Symbol)
    Community Gathering: 0 (Int64)
    SIC: 0 (Int64)
============================================================#
function add_household_structure!(Graph, nhouses)
    curr_num_v = nv(Graph)
    for house in (curr_num_v+1):(nhouses+curr_num_v)
        add_vertex!(Graph)
        set_props!(Graph, house, Dict(
                        :Type => :House,
                        :Community_Gathering => 0,
                        :SIC => 0))
    end
    return Graph
end

#============================================================
Adds businesses from Dataframe (Main_Sheet) to metagraph
(graph) where each business has the following property:
    Business_Name: (String)
    SIC: (Int64)?
    Max_Employees: (Int64)
    Employees: 0 (Int64)
    Type: Business (Symbol)
    business_type: (Int64) <-- (1 for public facing business)
============================================================#
function add_business_structure!(Graph, Main_Sheet)
    SIC_data = SIC_codes()
    Main_Sheet.SIC = lpad.(string.(Main_Sheet.SIC),6,"0")
    SIC_data.SIC_Sheet.SIC = lpad.(string.(SIC_data.SIC_Sheet.SIC),2,"0")
    select!(Main_Sheet,[:SIC,:EMPNUM])
    rename!(SIC_data.SIC_Sheet, "SIC" => "Short_SIC")
    transform!(Main_Sheet,:SIC => ByRow(x-> x[1:2]) => :Short_SIC)
    Main_Sheet = leftjoin(Main_Sheet, SIC_data.SIC_Sheet, on=:Short_SIC)
    #remove businesses without employees (ATMs & websites?)
    filter!(x -> x.EMPNUM != 0, Main_Sheet)

    #Make a metagraph
    curr_num_v = nv(Graph)
    for row in eachrow(Main_Sheet)
        add_vertex!(Graph)
        set_props!(Graph, rownumber(row)+curr_num_v,
                Dict(:Business_Name => "business_name",
                     :SIC => row.SIC,
                     :Max_Employees => row.EMPNUM,
                     :Employees => 0,
                     :Type => :Business,
                     :business_type => Vector(Main_Sheet[rownumber(row),5:end])))
    end
    return Graph
end

function jobs_open(g,v)
    get_prop(g,v,:Type) == :House && return false
    get_prop(g,v,:Max_Employees)-get_prop(g,v,:Employees) >0
end

function DataFrames.DataFrame(Businesses::XLSX.Worksheet)
    DataFrame(XLSX.gettable(Businesses)[2].=> XLSX.gettable(Businesses)[1])
end

## Add masks to the inhabitants of n=40 households
function add_masks!(model;n=40)
        agents = filter(x->x.home ≤ n, collect(allagents(model)))
        for agent in agents
                agent.mask = true
        end
end

#============================================================
------------------- Town Meta Functions ---------------------
SHOULD BE DEPRECATED WITH NEW AgentsIO API
============================================================#
function save_town(model,file)
    JLD.save(file,
            "agents", model.agents,
            "space", model.space,
            "properties", model.properties)
end

function load_town(file)
    town = JLD.load(file)
    model = ABM(Union{Adult,Child}, town["space"];
                properties = town["properties"],
                warn = false)
                for agent in town["agents"]
                    add_agent!(agent[2],model)
                end
    return model
end
