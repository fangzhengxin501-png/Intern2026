using POMCGraphSearch
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using LaserTag
using CSV, DataFrames
using Random 

Random.seed!(1)

rng = MersenneTwister(7)
pomdp = gen_lasertag(rng=rng, robot_position_known=false)


pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.3,
    max_search_depth = 100,
    num_fixed_observations=20,
    num_sim_per_sa = 3000,
    C_star = 1000
)

solve(pomcgs, pomdp)
ExportLogData(pomcgs, "LidarRoomba_log.csv")