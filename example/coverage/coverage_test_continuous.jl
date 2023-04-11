using PyPlot

using SubmodularMaximization
using Distributions
using LinearAlgebra
using Random


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

f(x) = mean_area_coverage(x, 100)
problem = ExplicitPartitionProblem(f, agents)

function evaluate_solver(solver,name)
  println("$name solver running")
  @time solution = solver(problem)

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

evaluate_solver(solve_continuous, "Continuous")



@show mean_weight(problem)
@show total_weight(problem)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

