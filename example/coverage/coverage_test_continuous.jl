using PyPlot

using SubmodularMaximization
using Distributions
using LinearAlgebra
using Random
using Base.Threads

pygui(false)
fig_path = "./fig/coverage_test"
mkpath(fig_path)

num_agents = 7
num_sensors = 7
nominal_area = 2.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = sensor_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

agents = generate_agents(agent_specification, num_agents)

f(x) = mean_area_coverage(x, 20)
problem = ExplicitPartitionProblem(f, agents)

function solve_continuous(p::PartitionProblem)
  num_agents = length(p.partition_matroid)
  num_sensors = length(p.partition_matroid[1].sensors)
  # Initialize y matrix
  y = zeros(num_sensors, num_agents)

  # Initialize t and delta
  t = 0.0
  delta = 1.0 / (num_agents * num_sensors)^2


  # Define a function to estimate the expected marginal profit of player i from item j
  function expected_marginal_profit(i, j, y, w, my_dict)
      # Generate a random set Ri containing each item j independently with probability yij(t)
      R = [x for x = 1:num_agents if x != j && rand() < y[i, x]]
      # if(length(R) != 0)
      #   print(R)
      # end
      
      g(j) = p.partition_matroid[j].sensors[i]
      R = map(g, R)
      newR = copy(R)
      push!(newR, p.partition_matroid[j].sensors[i])

      # Compute the expected marginal profit of player i from item j
      if haskey(my_dict, newR)
        wi_Ri_j =  my_dict[newR]
      else
        wi_Ri_j = w(newR)
        my_dict[newR] = wi_Ri_j
      end
     
      if haskey(my_dict, R)
        wi_Ri =  my_dict[R]
      else
        wi_Ri = w(R)
        my_dict[R] = wi_Ri
      end
      return wi_Ri_j - wi_Ri
  end
  # my_dict = Dict()
  # Run the algorithm
  while t < 1
      # Estimate the expected marginal profits for all players and items
      ω = zeros(num_sensors, num_agents)
      
      # for i = 1:num_sensors
      #     my_dict = Dict()
      #     for j = 1:num_agents
      #         for k = 1:(num_sensors * num_agents)^3
      #             ω[i, j] += expected_marginal_profit(i, j, y, p.objective, my_dict)
      #         end
      #         ω[i, j] /= (num_sensors * num_agents)^3
      #         # println("Loop2: $j")
      #     end
      # end
      
      @threads for i = 1:num_sensors
        my_dict = Dict()
        @threads for j = 1:num_agents
            ω_ij = 0.0
            for k = 1:(num_sensors * num_agents)^3
                ω_ij += expected_marginal_profit(i, j, y, p.objective, my_dict)
            end
            ω[i, j] = ω_ij / (num_sensors * num_agents)^3
        end
      end

      # Update y matrix
      for j = 1:num_agents
          i_star = argmax(ω[:, j])
          y[i_star, j] += delta
      end
      # Increment t
      t += delta
      println("Loop1: $t");
  end

  # Print the action distribution matrix
  println("Action distribution matrix:")
  println(y)
  
  selection = empty(p)
  for j = 1:length(p.partition_matroid)
      for i = 1:length(p.partition_matroid[1].sensors)
          if(rand() < y[i,j])
              push!(selection, (j,i))
              break
          end
      end
  end
  println(selection)
  evaluate_solution(p, selection)
end

function evaluate_solver(sol,name)
  println("$name solver running")
  @time solution = sol

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

sol = solve_continuous(problem)
evaluate_solver(sol, "Continuous")



@show mean_weight(problem)
@show total_weight(problem)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

