using POMCGraphSearch
using POMDPs
using Random
using POMDPTools
using RockSample

Random.seed!(1)

pomdp = RockSamplePOMDP(15, 15)  

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.4,
    max_search_depth = 40,
    num_sim_per_sa = 20,
    max_planning_secs = 36000.0,
    nb_particles = 50000,
    nb_sim_VMDP = 50000,
    epsilon_VMDP = 0.01
)

solve(pomcgs, pomdp)

ExportLogData(pomcgs, "RS15_log")