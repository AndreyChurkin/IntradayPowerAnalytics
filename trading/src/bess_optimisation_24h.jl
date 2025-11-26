"""
Formulates and solves the backtesting optimisation problem for BESS trading across 24 parallel and partially overlapping intraday trading sessions (corresponding to one full delivery day).

Inputs:
- DataFrame containing all bid and ask orders for up to 24 delivery hours of the day.
- BESS technical and economic parameters.

Outputs:
- The objective value (maximum achievable trading profit for the day).
- DataFrame with the optimal BESS trading actions across multiple parallel sessions (delivery hours within one day).

The model explicitly represents the battery's state of charge (SoC) over the real delivery times. 
Trading decisions are coupled across delivery hours via the SoC dynamics and power constraints.
Within a single delivery hour, trades can be interpreted as adjustments of a financial position within a single product. 
If the net traded volume for a given delivery hour is zero, such round-trip trades correspond to purely speculative arbitrage and do not impose any physical constraint on the battery. 
Only the net position per delivery hour is physically settled and is therefore subject to SoC and power constraints.

Andrey Churkin
https://andreychurkin.ru/

"""



function optimise_BESS_in_1_session(session_df::DataFrame;
    E_max::Float64,
    P_ch_max::Float64,  P_disch_max::Float64,
    eta_ch::Float64,    eta_disch::Float64,
    SoC_init::Float64,  SoC_final::Float64,
    E_cost_0,
    fee
    )

    Δt = 1.0 # consider only 1-hour physical delivery products

    T = nrow(session_df)

    bid_prices = session_df.bid_price
    ask_prices = session_df.ask_price
    bid_volumes = session_df.bid_volume
    ask_volumes = session_df.ask_volume

    # # Define the model and the variables:
    Model_24_sessions = Model(Gurobi.Optimizer)
    # set_silent(Model_24_sessions)
    @variable(Model_24_sessions, 0 <= ch[1:T] <= P_ch_max)
    @variable(Model_24_sessions, 0 <= disch[1:T] <= P_disch_max)
    @variable(Model_24_sessions, 0 <= SoC[1:T] <= E_max)
    @variable(Model_24_sessions, x_ch[1:T], Bin)
    @variable(Model_24_sessions, x_disch[1:T], Bin)

    # # Linking charging actions with volumes:
    @constraint(Model_24_sessions, [t=1:T], ch[t] <= P_ch_max * x_ch[t])
    @constraint(Model_24_sessions, [t=1:T], disch[t] <= P_disch_max * x_disch[t])

    # # Limiting actions by the ask/bid volumes available in the orders:
    @constraint(Model_24_sessions, [t=1:T], ch[t] <= ask_volumes[t])
    @constraint(Model_24_sessions, [t=1:T], disch[t] <= bid_volumes[t])

    # # Prevent simultaneous charge & discharge:
    @constraint(Model_24_sessions, [t=1:T], x_ch[t] + x_disch[t] <= 1)

    # # SoC dynamics:
    @constraint(Model_24_sessions, SoC[1] == SoC_init + (eta_ch*ch[1] - disch[1]/eta_disch) * Δt)
    @constraint(Model_24_sessions, [t=2:T], SoC[t] == SoC[t-1] + (eta_ch*ch[t] - disch[t]/eta_disch) * Δt)
    @constraint(Model_24_sessions, SoC[T] == SoC_final)

    # # Objective:
    @objective(Model_24_sessions, Max, SoC_init*E_cost_0 
                + sum(((bid_prices[t] - fee)*disch[t] - (ask_prices[t] + fee)*ch[t]) * Δt for t in 1:T)
    )

    optimize!(Model_24_sessions)


    if termination_status(Model_24_sessions) == MOI.OPTIMAL
        printstyled("\n✅ Optimal solution found, objective value = ", objective_value(Model_24_sessions), color = :green)
    else
        printstyled("\n❌ WARNING: Solver did not return an optimal solution. Status = ", termination_status(Model_24_sessions), color = :red)
    end

    BESS_optimisation_results = DataFrame(
        ts = session_df.ts,
        hours_to_delivery = session_df.hours_to_delivery,
        ask_price = session_df.ask_price,
        bid_price = session_df.bid_price,
        action_ch = Int.(value.(ch) .> 1e-6),
        action_disch = Int.(value.(disch) .> 1e-6),
        volume_ch = value.(ch),
        volume_disch = value.(disch),
        SoC = value.(SoC),
        cum_revenue = cumsum((value.(disch) .* session_df.bid_price) - (value.(ch) .* session_df.ask_price))
    )

    return objective_value(Model_24_sessions), BESS_optimisation_results

end