using AdaOPS
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using RockSample
using ParticleFilters
using CSV, DataFrames
using Random
using Dates

Random.seed!(1)

pomdp = RockSamplePOMDP(15, 15)

bounds = AdaOPS.IndependentBounds(
    FORollout(RSExitSolver()),
    FOValue(RSMDPSolver()),
    check_terminal = true,
    consistency_fix_thresh = 1e-5
)

planning_time = 3.0   # equivalent of POMCGS's `planning_time`: per-STEP online budget (sec)
max_depth = 100        # equivalent of POMCGS's `max_depth`: episode length cap (steps)

# --- Rough parameter correspondence with your POMCGS solver, for reference:
#   max_b_gap (POMCGS belief-merge threshold ξ)   ~= delta (AdaOPS L1 packing threshold)
#   max_search_depth (POMCGS)                     ~= max_depth field below (bound-rollout depth, NOT episode length)
#   num_sim_per_sa (POMCGS)                        -- no direct analog; closest is m_min/m_max (particle counts per belief)
# These are NOT numerically equivalent parameters (different algorithms), just the closest
# conceptual match — don't assume the same numeric value means "the same setting".
adaops_solver = AdaOPSSolver(
    bounds = bounds,
    T_max = planning_time,   # <-- this is the field that actually caps online planning time per step
    delta = 0.3,
    m_min = 30,
    m_max = 200,
    num_b = 10000,
    tree_in_info = false
)

adaops = solve(adaops_solver, pomdp)

nb_runs = 100
n_particles = 10000   # particle filter used to TRACK THE TRUE BELIEF during simulation
                        # (separate from AdaOPS's internal search-tree particles above)

results = Float64[]

for i in 1:nb_runs
    b0 = initialstate(pomdp)
    s0 = rand(b0)
    up = BootstrapFilter(pomdp, n_particles)

    # RolloutSimulator returns the (discounted) episode return, same quantity
    # POMCGraphSearch.SolveOnline returned as `run_return`.
    run_return = simulate(RolloutSimulator(max_steps = max_depth), pomdp, adaops, up, b0, s0)
    push!(results, run_return)
    println("Run $i: $run_return")
end

# Save results to CSV
df = DataFrame(run_id = 1:nb_runs, return_value = results)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
CSV.write("RS15_AdaOPS_$(mean(results))_$(timestamp).csv", df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results))")
println("Results saved to RS15_AdaOPS_$(timestamp).csv")