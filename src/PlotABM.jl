#============================================================
---------------------- Main Functions -----------------------
============================================================#
function plot_grid(EpidemicInvariantsParentDirectory; MaskingLevels=5, VaccineLevels=5, NumberRunsPerLevel=100)
    # Compute Masking and Vaccine levels
    mask_incr = floor(100/MaskingLevels)
    vacc_incr = floor(100/VaccineLevels)

    # Initialize the grid of subplots
    Grid_of_Plots = make_subplots(
        rows = 5, cols = 5,
        vertical_spacing = 0.02,
        horizontal_spacing = 0.02,
        x_title = "Masking Level",
        y_title = "Vaccine Level",
        column_titles = ["0%", "20%", "40%", "60%", "80%"],
        row_titles = ["0%", "20%", "40%", "60%", "80%"]
    )

    # Create and add subplots to the grid
    Grid_Col = 1
    for mask_lvl in 0:mask_incr:99
        Grid_Row = 1
        for vacc_lvl in 0:vacc_incr:99
            Scatter_Data = []
            for run in 1:NumberRunsPerLevel
                # Import data
                EpidemicInvariantsFilePath = "$(EpidemicInvariantsParentDirectory)$(@sprintf("%.2d",Int(mask_lvl)))_$(@sprintf("%.2d",Int(vacc_lvl)))"
                EpidemicInvariantsFilePathNumber = "$(EpidemicInvariantsFilePath)_$(@sprintf("%.3d",run))_data.csv"
                InvariantsDF = CSV.File(EpidemicInvariantsFilePathNumber) |> DataFrame

                # Build Plots
                add_trace!(
                    Grid_of_Plots,
                    scatter(
                        x=InvariantsDF[!,1],
                        y=InvariantsDF[!,"Symptomatic"],
                        mode = "lines",
                        line_color = "#212121",
                        opacity = 0.1,
                    ),
                    row = Grid_Row,
                    col = Grid_Col
                )
            end
            Grid_Row += 1
        end
        Grid_Col += 1
    end

    # Update features of each subplot axis
    update_yaxes!(Grid_of_Plots, range=[0,300], dtick=150, tickfont=attr(size = 9))
    update_xaxes!(Grid_of_Plots, range=[0,120], dticks=120, tickfont=attr(size = 9))

    # Update grid layout properties
    relayout!(
        Grid_of_Plots,
        showlegend = false,
        title = attr(text="Symptomatic Agents over Time", x=0.5, font=attr(size=28)),
        margin = attr(l=70, r=30, b=70)
    )

    return Grid_of_Plots
end

function plot_epidemic_invariants(EpidemicInvariantsFilePath)
    # Import data
    InvariantsDF = CSV.File(EpidemicInvariantsFilePath) |> DataFrame

    # Compute Statistics
    total_infected = InvariantsDF[end,2] + InvariantsDF[end,3] + (InvariantsDF[1,4] - InvariantsDF[end,4]) # Current_Infected + Recovered + Dead
    total_recovered = InvariantsDF[end,3]
    max_infected, peak_day = findmax(InvariantsDF.Symptomatic)
    peak_day -= 1

    # Build Plots
    graph_data = AbstractTrace[]
    for col in names(InvariantsDF)[2:end]
        push!(graph_data, scatter(x=InvariantsDF[!,1], y=InvariantsDF[!,col], mode = "lines", name = col))
    end

    max_day_data = scatter(;x=[peak_day], y=[max_infected],
                      mode="markers+text", name="Peak Day",
                      textposition="top center",
                      text=["Peak Day: $(max_infected) infected"],
                      marker_size=7, textfont_family="Raleway, sans-serif")

    push!(graph_data, max_day_data)

    return Plot(graph_data,
                Layout(
                    title = attr(
                        text = "Infection Over Time",
                        x=0.45,
                        xanchor= "center",
                        yanchor= "top"
                    ),
                    xaxis_title = "Days",
                    yaxis_title = "Case Count",
                    paper_bgcolor= "#212121",
                    plot_bgcolor= :transparent,
                    font_color = "D4D4D4",
                    height = 600,
                    width = 700
                    )
                )
end

#============================================================
--------------------- Helper Functions ----------------------
============================================================#
function save_epidemic_invariants_plot(EpidemicInvariantsFilePath::String, savepath::String)
    savefig(plot_epidemic_invariants(EpidemicInvariantsFilePath), savepath)
end

function save_plot_grid(EpidemicInvariantsParentDirectory; SaveFilePath = "grid.png",Height=1000, Width=1000, MaskingLevels=5, VaccineLevels=5, NumberRunsPerLevel=100)
    savefig(
        plot_grid(
            EpidemicInvariantsParentDirectory,
            MaskingLevels=MaskingLevels,
            VaccineLevels=VaccineLevels,
            NumberRunsPerLevel=NumberRunsPerLevel
            ),
        height=Height,
        width=Width,
        SaveFilePath)
end
