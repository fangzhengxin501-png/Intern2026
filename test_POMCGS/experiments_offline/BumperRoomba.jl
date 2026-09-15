using POMCGraphSearch
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using CSV, DataFrames
using Random 
using RoombaPOMDPs


Random.seed!(1)

num_x_pts, num_y_pts, num_th_pts = 25, 16, 10
sspace = DiscreteRoombaStateSpace(num_x_pts, num_y_pts, num_th_pts)

max_speed, speed_interval = 5.0, 0.2
max_turn_rate, turn_rate_interval = 1.0, 0.2
action_space = vec([RoombaAct(v, ω)
                    for v in 0:speed_interval:max_speed,
                        ω in -max_turn_rate:turn_rate_interval:max_turn_rate])

pomdp = RoombaPOMDP(sensor=Bumper(),
    mdp=RoombaMDP(config=3, aspace=action_space,
    v_max=max_speed, sspace=sspace))

pomcgs = SolverPOMCGS(pomdp;
                max_search_depth = 80,
                max_b_gap = 0.05,
                bool_APW = false,
                C_star = 1000,
                num_sim_per_sa = 1000,
                max_planning_secs = 20000.0
)

solve(pomcgs, pomdp)
ExportLogData(pomcgs, "BumperRoomba_log.csv")
SavePrunedPolicyJSON(pomcgs.fsc, outfile_name="BumperRoomba_policy")
SavePrunedPolicyDOT(pomcgs.fsc, outfile_name="BumperRoomba_policy")