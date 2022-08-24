function plot_adj_matrix(model, min_w)
    # Generate graph object from adjacency matrix
    M = Get_Adjacency_Matrix(model)
    G = graph_from_adj(M)

    # Filter out edges with weight < min_w
    for edge in collect(edges(G))
        if M[src(edge),dst(edge)] < min_w
            rem_edge!(G, edge)
        end
    end

    # Apply layout algorithm to graph
    pos_x, pos_y = spring_layout(G,5.0)

    # Create plot points
    edge_x = []
    edge_y = []

    for edge in edges(G)
        push!(edge_x, pos_x[src(edge)])
        push!(edge_x, pos_x[dst(edge)])
        push!(edge_y, pos_y[src(edge)])
        push!(edge_y, pos_y[dst(edge)])
    end

    # Initialize graph data vector
    graph_data = GenericTrace{Dict{Symbol, Any}}[]

    # Find max weight in matrix
    max_w = 0.0
    num_agents = Int(sqrt(length(M)))
    for i in 1:num_agents-1
        for j in i+1:num_agents
            temp_w = M[i,j]
            if max_w < temp_w
                max_w = temp_w
            end
        end
    end

    # Create edges
    for edge in edges(G)
        temp_trace = PlotlyJS.scatter(
            mode="lines",
            opacity = M[src(edge),dst(edge)]/max_w,
            x=[pos_x[src(edge)], pos_x[dst(edge)]],
            y=[pos_y[src(edge)], pos_y[dst(edge)]],
            line=attr(
                width=1.0,
                color="#FFF"
            )
        )
        push!(graph_data, temp_trace)
    end

    # Create nodes
    nodes_trace = PlotlyJS.scatter(
        x=pos_x,
        y=pos_y,
        mode="markers",
        marker=attr(
            size=4,
            color="00FFFF"
        )
    )
    push!(graph_data, nodes_trace)



    # Create plot object
    return graph_data
end

function get_connected_components(model;min_w = 0)
    M = Get_Adjacency_Matrix(model)
    G = graph_from_adj(M,min_w= min_w)
    return connected_components(G)
end

function graph_from_adj(M; min_w = 0)
    # Intialize variables
    num_agents = Int(sqrt(length(M)))
    G = SimpleGraph(num_agents)

    # Create Graph object
    for i in 1:num_agents-1
        for j in i+1:num_agents
            if M[i,j] >= min_w
                add_edge!(G, i, j)
            end
        end
    end

    return G
end

function spring_layout(g::AbstractGraph, k;
    temp = 4.0,
    iters = 60)

    # Variables
    MIN_ATT_DIST = 3.5
    MAX_REP_DIST = 10.0
    TEMP_MIN = 0.5

    # Comput useful constants
    nvg = nv(g)
    adj_mat = adjacency_matrix(g)

    # Position nodes evenly along circle
    pos_x = zeros(nvg)
    pos_y = zeros(nvg)
    for i in 1:nvg
        pos_x[i] = 10*cos(i/nvg * 6.28)
        pos_y[i] = 10*sin(i/nvg * 6.28)
    end

    # Main loop of Fruchterman-Reingold layout algorithm
    for i in 1:iters
        # Initialize displacement vectors
        disp_x = zeros(nvg)
        disp_y = zeros(nvg)

        # loop over all nodes
        for v in 1:nvg-1
            for u in v+1:nvg
                # Skip if u == v
                u == v && continue

                # Compute distance between nodes
                delta_x = pos_x[v] - pos_x[u]
                delta_y = pos_y[v] - pos_y[u]
                distance = norm(delta_x,delta_y)

                # Compute repulsion force for neighboring nodes
                if distance < MAX_REP_DIST
                    # Repulsion force
                    rep = (k^2)/distance
                    scaler_x = delta_x / distance * rep
                    scaler_y = delta_y / distance * rep

                    # Apply to v
                    disp_x[v] += scaler_x
                    disp_y[v] += scaler_y

                    # Appy to u
                    disp_x[u] -= scaler_x
                    disp_y[u] -= scaler_y
                end

                # Compute attractive force
                if isone(adj_mat[v,u])
                    # Ignore nodes which are close enough
                    distance < MIN_ATT_DIST && continue

                    # Attraction force
                    att = distance^2 / k
                    scaler_x = delta_x / distance * att
                    scaler_y = delta_y / distance * att

                    # Apply to v
                    disp_x[v] -= scaler_x
                    disp_y[v] -= scaler_y

                    # Appy to u
                    disp_x[u] += scaler_x
                    disp_y[u] += scaler_y
                end
            end
        end

        # Limit net movement by temperature
        capped_disp_x = zeros(nvg)
        capped_disp_y = zeros(nvg)
        for v in 1:nvg
            disp_norm = norm(disp_x[v], disp_y[v])

            # Ignore overlapping nodes (unlikely to occur)
            disp_norm < 0.0001 && continue
            max_disp = min(disp_norm, temp)

            # Compute capped displacement
            capped_disp_x[v] = disp_x[v] / disp_norm * max_disp
            capped_disp_y[v] = disp_y[v] / disp_norm * max_disp

            # Apply forces to node
            pos_x[v] += capped_disp_x[v]
            pos_y[v] += capped_disp_y[v]
        end

        # Cool down the temperature
        temp > TEMP_MIN ? temp *= 0.9 : nothing
    end

    return pos_x, pos_y
end

function norm(a,b)
    return sqrt(a^2 + b^2)
end
