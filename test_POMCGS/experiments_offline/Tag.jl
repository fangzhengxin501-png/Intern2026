using POMCGraphSearch
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using TagPOMDPProblem
using CSV, DataFrames
using Random 

Random.seed!(1)


pomdp = TagPOMDP()

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.13,
    max_search_depth = 60,
    num_sim_per_sa = 10000
)


solve(pomcgs, pomdp)
ExportLogData(pomcgs, "Tag_log")
SavePrunedPolicyJSON(pomcgs.fsc, outfile_name="Tag_policy")
SavePrunedPolicyDOT(pomcgs.fsc, outfile_name="Tag_policy")
