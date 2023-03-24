import numpy as np
import pandas as pd

from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.graphs import GraphGeodesicDistance

homology_dimensions = [0, 1, 2]
adjacencyMatrixList = []
input_fn = "RipVSGiotto.csv"
metric = 'precomputed'

# Import ABM adjacency matrix
adjacency_matrix = pd.read_csv(input_fn)
adjacency_matrix = adjacency_matrix.to_numpy()
adjacency_matrix = np.divide(
    np.ones(
        (386, 386)
    ),
    adjacency_matrix,
    out=np.ones_like(adjacency_matrix)*np.inf,
    where=adjacency_matrix != 0
    )

# Making the diagonals zeros instead of np.inf
np.fill_diagonal(adjacency_matrix, 0)
adjacency_matrix = np.ascontiguousarray(adjacency_matrix)

adjacencyMatrixList.append(adjacency_matrix)

#Make the distance the shortest path
X_ggd = GraphGeodesicDistance(directed=False, unweighted=False).fit_transform(adjacencyMatrixList)
# Compute the persistence diagrams
VR = VietorisRipsPersistence(
    homology_dimensions=homology_dimensions,
    metric=metric
)
diagrams = VR.fit_transform(X_ggd)

#Can use this chunk to dynamically find step size. I think it will speed thinks up to not           
birth_death_df=pd.DataFrame(diagrams[0])
epsilon=10**(-5)

# Group features by dimension
birth_death_df0 = birth_death_df.loc[birth_death_df[2] == 0.0]
birth_death_df1 = birth_death_df.loc[birth_death_df[2] == 1.0]
birth_death_df2 = birth_death_df.loc[birth_death_df[2] == 2.0]

# Starting at the minimum non-zero birth, increase the distance by epsilon and count the number of features of each dimension; storing the results in a dataframe (ranks)
# This process will create a time (epsilon) series on the counts of features grouped by dimension
dist = min(i for i in birth_death_df[0] if i > 0.0)
ranks = pd.DataFrame(columns=['dist', 'h0', 'h1', 'h2'])
while dist < 0.115:
    count_birth_death_df0 = birth_death_df0.loc[(
        birth_death_df0[0] <= dist) & (dist <= birth_death_df0[1])]
    count_birth_death_df1 = birth_death_df1.loc[(
        birth_death_df1[0] <= dist) & (dist <= birth_death_df1[1])]
    count_birth_death_df2 = birth_death_df2.loc[(
        birth_death_df2[0] <= dist) & (dist <= birth_death_df2[1])]
    running_birth_death_df = pd.DataFrame({
                                'dist': [dist],
                                'h0': [len(count_birth_death_df0)],
                                'h1': [len(count_birth_death_df1)],
                                'h2': [len(count_birth_death_df2)],
                                'sum': [len(count_birth_death_df1)+len(count_birth_death_df2)]})
    ranks = pd.concat([ranks, running_birth_death_df], ignore_index=True)
    dist = dist+epsilon

# Remove rows where the sum of features is less than 3
ranks_filtered = ranks.loc[ranks['sum'] >= 3]
ranks_filtered = ranks_filtered.reset_index(drop=True)
print(ranks)
print(ranks_filtered)

# Compute the thickness at each epislon step
thickness_dist = pd.DataFrame({'tau': [((ranks_filtered['h2'][i]-ranks_filtered['h1'][i])/(
    ranks_filtered['h1'][i]+ranks_filtered['h2'][i])) for i in range(len(ranks_filtered))]})
thickness_df=pd.DataFrame(columns=range(1,101))
thickness_df[1]=thickness_dist['tau']
        
        
thickness_df.to_csv(
        f'taudist.csv'
    )