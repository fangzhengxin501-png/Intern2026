using POMCGraphSearch
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using CSV, DataFrames
using Random 
using RoombaPOMDPs


Random.seed!(1)

num_x_pts, num_y_pts, num_th_pts = 25, 16, 10
sspace = DiscreteRoombaStateSpace(num_x_pts, num_y_pts, num_th_pts)

# Define a large discrete action set for continuous action approximation
max_speed, speed_interval = 5.0, 0.2
max_turn_rate, turn_rate_interval = 1.0, 0.2
action_space = vec([RoombaAct(v, ω)
                    for v in 0:speed_interval:max_speed,
                        ω in -max_turn_rate:turn_rate_interval:max_turn_rate])

pomdp = RoombaPOMDP(sensor=Lidar(),
    mdp=RoombaMDP(config=3, aspace=action_space,
    v_max=max_speed, sspace=sspace))

pomcgs = SolverPOMCGS(pomdp;
    max_search_depth = 50,
    max_b_gap = 0.15,
    bool_APW = false,
    num_fixed_observations = 10
)

solve(pomcgs, pomdp)
ExportLogData(pomcgs, "LidarRoomba_log")
SavePrunedPolicyJSON(pomcgs.fsc, outfile_name="LidarRoomba_policy")
SavePrunedPolicyDOT(pomcgs.fsc, outfile_name="LidarRoomba_policy")