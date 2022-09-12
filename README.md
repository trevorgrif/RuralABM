# RuralABM
An agent based model for epidemic simulation in rural towns with adjustable social behaviors.

## Loading into Julia
Loading the RuralABM into a julia session can be done in two ways. After downloading the code base open a julia session and in the `Pkg` REPL (press `]` to access) run

```julia
pkg> develop Path/To/RuralABM
```

Alternatively, RuralABM can be loaded while opening the julia session with

```cmd
julia -L Path/To/RuralABM.jl
```

In either case, RuralABM can be brought into scope with

```julia
julia> using RuralABM
```

Even though this will bring all methods into scope, the only exported methods can be found in `src/interface.jl`.
## Running the Model
Before running the simulation, a town must be constructed with the `Construct_Town` method which takes two CSV files as input. The first CSV file contains information on the population: agent ID, household assignment, age range, and income. The second CSV details the businesses structure: SIC code and number of employees.

Included in the codebase are two example towns: a small town with 386 agents and 62 businesses, and a larger town with 5,129 agents and 353 businesses.

`Construct_Town` will return an Agents.jl model object and three dataframes detailing counts on various model objects. See the documentation for further details on the dataframes returned.

The model object can than be passed to `Seed_Contagion!` which defaults to one seeding, but can be altered using the parameter `seed_num`.

The epidemic will now spread (or die out) by calling `Run_Model!(model)`, which will continue to run the model until there are no infected agents remaining. Optionally, the simulation can be run for a fixed number of days with the parameter `duration`.

Similar to `Construct_Town`, `Run_Model!` will return the model object along with the adata (see `Agents.jl`), the transmission network, the social contact matrix, and some summary statistics of the epidemic.

## Agent Structure
See `src/Structs.jl` for specific data structures held within each agent type. There are three agents in the model, `Child`, `Adult`, and `Retiree`. Most attributes are identical across each agent type with the exception of:

 - Only adults have a job, a shift, and an income assigned
 - Instead of work, children have a school assigned
 - Adults and retirees may have a community gathering assigned, whereas children attend the community gathering of their parents

## Town Structure
The town has some core building types: houses, businesses, schools, and community gatherings. Business mappings are determined by SIC code and can be altered in `src/Town.jl`.

## Masking and Vaccinations
Each agent has `bool::masked` and `Vector{bool}::will_mask` parameters. The former changes based on the location and context of the agent. Conceptually, this flag determines if the agent is actively wearing a mask. The ladder defines the context in which an agent *will* wear a mask.  Each flag in `will_mask` vector describes the following:

 - `will_mask[1]` (Global): Agent will wear a mask when leaving their scheduled location to socialize with another agent (e.g. visiting a friend's home or workplace, going shopping, ect.)
 - `will_mask[2]` (Local): Agent will wear a mask while socializing with another agent within their scheduled location (e.g. at work, at home, ect.)
 - `will_mask[3]` (Social): Agent will wear a mask when explicitly leaving their location to interact with a **friend**

Vaccinating an agent is as simple as changing their `:status` to `:V`. Since the status of an agent may change throughout the epidemic, it is also recommended to change the agents `vaccinated` flag to `true` for recording epidemic statistics.

Agent attributes can be updated with

```julia
Update_Agents_Attribute!(model, agent_id_array, attribute_to_change, new_value)
```

Selecting target agents can either be done manually, randomly with `Get_Portion_Random`, or by a cascading model with `Get_Portion_Watts`.

## Model Step
Each model step increments the hour by one. The time structure of the model is five weekdays where agents attend work and school, and one weekend day where agents may attend a community gathering. Each day is twelve hours long.

## Agent Step
Each hour, every agent makes a decision on what action to take. Although the likelihood of making certain decision varies dynamically by agent type, location, time of day, and time of week, generally an agent will do one of five things:

 - Socialize Globally: Interact with a random agent at a different location
 - Socialize Locally: Interact with a random agent at their current location
 - Hang with a Friend: Go to a friends location and interact
 - Go Shopping: Go to a random business and interact with an agent there
 - Do Nothing: Remain at their location and do nothing

 Distributions on likelihood of each action per agent type can be altered in `src/Structs.jl`.

## Disease Spread

During agent interactions, if one of the agents is infected, then there is a possibility for disease transmission. Transmission can be thought of as a Bernoulli random variable with the probability of transmission given by $p(t) = C \gamma(t;\alpha,\beta)$ where $\gamma(t;\alpha,\beta)$ is the probability density function for the gamma distribution with shape $\alpha$ and scale $\beta$, and $C$ is a dimensionless constant. The shape and scale were taken from [^1], and the constant $C$ was determined so that the expected number of new infections approximately matches estimates of the basic reproductive number $\mathcal{R}_0$ for COVID-19. Once the probability of transmission falls below a certain threshold, we consider the agent `recovered', at which point they are immune to re-infection.

## Model Parameters
Lots... See `src/Structs.jl`

[^1]: Xi He, Eric HY Lau, Peng Wu, Xilong Deng, Jian Wang, Xinxin Hao, Yiu Chung Lau,
Jessica Y Wong, Yujuan Guan, Xinghua Tan, et al. <i>Temporal dynamics in viral shedding
and transmissibility of covid-19.</i>, Nature medicine, 26(5):672–675, 2020