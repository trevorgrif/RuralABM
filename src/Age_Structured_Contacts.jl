function SampleContacts(model, subset)
    Contacts = DataFrame(agent = Int64[], contact = Int64[] )
    for agent in subset
        for contact in findall(!iszero, model[agent].contact_list)
            push!(Contacts,[agent, contact])
        end
    end
    return Contacts
end

function ContactMatrix(model,subset)
    Contacts = SampleContacts(model,subset)
    contact_age = select(Population,[:agent,:age])
    Contacts = leftjoin(Contacts, contact_age, on = :agent)
    rename!(Contacts, :age => :agent_age)
    rename!(contact_age,:agent => :contact)
    Contacts = leftjoin(Contacts, contact_age, on = :contact)
    rename!(Contacts, :age => :contact_age)
    Totals = combine(groupby(Contacts,[:agent,:agent_age]),nrow => :Number_of_contacts)
    Totals = combine(groupby(Totals,:agent_age),nrow => :Size_of_age_group)
    Contacts = combine(groupby(Contacts,[:agent_age,:contact_age]),nrow => :number)
    Contacts = leftjoin(Contacts,Totals, on = :agent_age)
    transform!(Contacts,[:number,:Size_of_age_group] => ((x,y)-> x./y) => :average)
    select!(Contacts,[:agent_age,:contact_age,:average])
    Contacts = unstack(Contacts,:agent_age,:contact_age,:average)
    Contacts = Matrix(coalesce.(Contacts, 0.0)[:,2:end])
end


## After building the town, make a spreadsheet with information about each agent
Grangeville = build_town("output/abm_cityoutput_Grangeville_ID_10_mile_income_ind.csv")
Population = demographics(Grangeville)
# Run simulation without disease dynamics for 1 day
subset = shuffle(1:nrow(Population))


ndays = 14

step!(Grangeville,agent_step!,model_step!,12*ndays)
CMat = ContactMatrix(Grangeville,subset)./ndays

using Plots
using ColorBrewer
plt = heatmap(CMat, c = cgrad([:darkblue,:green,:yellow,:white]))
xlabel!("Contact age")
ylabel!("Participant age")

savefig(plt,"output/CMat_2021-02-22.png")
