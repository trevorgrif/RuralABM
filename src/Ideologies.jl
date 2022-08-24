## Testing out ideologies based on voting data. See onenote for more info
counties = CSV.File("countypres_2000-2020.csv") |> DataFrame
# Filter data to just Idaho County, Idaho
data = filter(x->x.county_fips == "24033", counties)
filter!(x->x.year == 2020,data)

data.candidatevotes = parse.(Float64,data.candidatevotes)

function score(data)
    liberal = filter(x->x.party == "DEMOCRAT",data).candidatevotes |> sum
    other = filter(x->x.party ∈ ["GREEN","OTHER"],data).candidatevotes |> sum
    conservative = filter(x->x.party == "REPUBLICAN",data).candidatevotes |> sum
    libertarian = filter(x->x.party == "LIBERTARIAN",data).candidatevotes |> sum
    E = (0.3liberal+0.7conservative+libertarian)/(sum(data.candidatevotes))
    V = (liberal*(0.3-E)^2+other*(E)^2+conservative*(0.7-E)^2+libertarian*(1-E)^2)/(sum(data.candidatevotes)-1)
    α = (E*(1-E)/V-1)*E
    β = (E*(1-E)/V-1)*(1-E)
    Beta(α,β)
end


ideology = score(data)

plot(0:0.001:1,x->pdf(ideology,x))
