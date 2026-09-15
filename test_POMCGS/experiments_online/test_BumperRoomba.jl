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
                max_search_depth = 50,
                max_b_gap = 0.05,
                bool_APW = false,
                num_sim_per_sa = 500
)

planning_time = 3.0
max_depth = 100
nb_runs = 100 

results = Float64[]

for i in 1:nb_runs
    run_return = POMCGraphSearch.SolveOnline(pomcgs, max_depth, planning_time; verbose=true)
    push!(results, run_return)
    println("Run $i: $run_return")
end

# Save results to CSV
df = DataFrame(run_id = 1:nb_runs, return_value = results)
CSV.write("pomcgs_results_BumperRoomba.csv", df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results))")
println("Results saved to pomcgs_results.csv")