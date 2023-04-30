using PyPlot

using SubmodularMaximization

pygui(false)
fig_path = "./fig/coverage_test"
mkpath(fig_path)

num_agents = 5
num_sensors = 5
nominal_area = 2.0


sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = sensor_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

agents = generate_agents(agent_specification, num_agents)

f(x) = mean_area_coverage(x, 50)
# agents[1].sensors[1].center = [0.4,0.6]
# agents[1].sensors[1].radius = 0.4
# agents[1].sensors[2].center = [0.0,0.0]
# agents[1].sensors[2].radius = 0.4
# agents[1].sensors[3].center = [-0.2,0.0]
# agents[2].sensors[1].center = [0.35,0.6]
# agents[2].sensors[1].radius = 0.4
# agents[2].sensors[2].center = [-0.3,0.0]
# agents[2].sensors[2].radius = 0.4
# agents[2].sensors[3].center = [0.4,-0.2]
# agents[3].sensors[1].center = [0.4,0.1]
# agents[3].sensors[2].center = [0.8,0.8]
# agents[3].sensors[3].center = [0.5,0.5]
problem = ExplicitPartitionProblem(f, agents)
# println(agents[1].sensors[1])
# println(agents[1].sensors[2])
# println(agents[1].sensors[3])

function evaluate_solver(solver, name)
  println("$name solver running")
  if(name == "Multiround Sequential")
    @time solution = solver(problem, 3)
  else
    @time solution = solver(problem)
  end
  # @time solution = solver(problem)

  figure()
  xlim([0, 1])
  ylim([0, 1])
  colors = generate_colors(agents)
  visualize_agents(agents, colors)
  visualize_solution(problem, solution, colors)

  gca().set_aspect("equal")
  savefig("$(fig_path)/$(to_file(name)).png", pad_inches=0.00, bbox_inches="tight")

  coverage = solution.value

  title("$name Solver ($coverage)")

  @show coverage
end

#evaluate_solver(solve_optimal, "Optimal")
#evaluate_solver(solve_worst, "Worst-case")
#evaluate_solver(solve_myopic, "Myopic")
#evaluate_solver(solve_random, "Random")
evaluate_solver(solve_sequential, "Sequential")
evaluate_solver(solve_continuous, "Continuous")
evaluate_solver(solve_sequential_multiround, "Multiround Sequential")

# for num_partitions in [2, 4, 8]
#   solve_n(p) = solve_n_partitions(num_partitions, p)
#   evaluate_solver(solve_n, "Partition-$num_partitions")
# end

#@show mean_weight(problem)
#@show 



total_weight(problem)

