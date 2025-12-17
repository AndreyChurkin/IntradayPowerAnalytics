"""
This code analyses and plots the optimal trading actions within one trading session only (for one delivery hour).
It will later be extended to consider 24 parallel trading sessions.

Andrey Churkin
https://andreychurkin.ru/

"""



cd(dirname(@__FILE__))
println(pwd())

using CSV
using DataFrames
using Dates
using JuMP, Gurobi
using Suppressor
using Plots, Plots.PlotMeasures
using Statistics



# # Select the intraday market data set to analyse:
ID_market_data_file_path = "C://Users//achurkin//Documents//MEGA//Imperial College London//Pierre Pinson//models//IDC_EPEX_DK1_BestBidAsk_clean_v1.csv"


ID_market_data = CSV.read(ID_market_data_file_path, DataFrame)
# ID_market_data = CSV.read(ID_market_data_file_path, DataFrame, limit = 2*10^6) # <--- read a limited data set to save time during testing


# # Define the trading session to optimise (delivery hour):
delivery_hour_to_optimise = "2023-01-01 01:00:00"
# delivery_hour_to_optimise = "2023-01-02 01:00:00"
# delivery_hour_to_optimise = "2023-12-30 15:00:00" # why no profit? mistake?
# delivery_hour_to_optimise = "2023-06-25 20:00:00" # high profit, few actions
# delivery_hour_to_optimise = "2023-05-28 11:00:00" # high profit, a lot of actions
# delivery_hour_to_optimise = "2023-05-28 12:00:00" # high profit, a lot of actions
# delivery_hour_to_optimise = "2023-07-02 13:00:00" # high profit, a lot of actions. -275EUR prices? why??
# delivery_hour_to_optimise = "2023-05-28 14:00:00" # high profit, a lot of actions. negative prices? why??
# delivery_hour_to_optimise = "2023-05-28 15:00:00" # +-
# delivery_hour_to_optimise = "2023-08-07 12:00:00" # +-
# delivery_hour_to_optimise = "2023-08-08 02:00:00"



# # Remove orders in the records that are for sessions beyond the delivery hour:
ID_market_data = filter(:delivery_start => ==(delivery_hour_to_optimise), ID_market_data)


# # Remove duplicating rows (same prices/orders for the same millisecond timestamps):
ID_market_data = unique(ID_market_data, [:ts, :delivery_start, :ask_price, :ask_volume, :bid_price, :bid_volume])


# # Sort by timestamp to ensure correct chronological trades:
ID_market_data = sort(ID_market_data, :ts)


# # Convert timestamps of orders into hours till delivery time (for further plotting):
hours_to_delivery = (Dates.value.(
                    DateTime.(ID_market_data.ts, "yyyy-mm-dd HH:MM:SS.sss") 
                    .- 
                    DateTime.(delivery_hour_to_optimise, "yyyy-mm-dd HH:MM:SS.sss")
                    ) / (60 * 60 * 1000)
)

ID_market_data.hours_to_delivery = hours_to_delivery

# # Remove repeating orders from successive timestamps by nullifying their volumes (needed to make BESS backtesting more realistic)
successive_ask_order_removals_count = 0
successive_bid_order_removals_count = 0
ID_market_data_copy_all_volumes = copy(ID_market_data)
for i in 2:nrow(ID_market_data)
    if ID_market_data.ask_price[i] == ID_market_data_copy_all_volumes.ask_price[i-1] && ID_market_data.ask_volume[i] == ID_market_data_copy_all_volumes.ask_volume[i-1]
        ID_market_data.ask_volume[i] = 0.0
        global successive_ask_order_removals_count += 1
    end
    if ID_market_data.bid_price[i] == ID_market_data_copy_all_volumes.bid_price[i-1] && ID_market_data.bid_volume[i] == ID_market_data_copy_all_volumes.bid_volume[i-1]
        ID_market_data.bid_volume[i] = 0.0
        global successive_bid_order_removals_count += 1
    end
end

println()
println("Trading session to be optimised for delivery hour ",delivery_hour_to_optimise)
println("Total number of best bid/ask price changes = ",nrow(ID_market_data))
println(successive_ask_order_removals_count, " successive Ask orders with the same price/volume have been removed")
println(successive_bid_order_removals_count, " successive Bid orders with the same price/volume have been removed")
println("The earliest order placed: ",minimum(hours_to_delivery)," h to delivery")
println("The latest order placed: ",maximum(hours_to_delivery)," h to delivery")
println()




# # Read the day-ahead market data:
DA_market_data_file_path = "C://Users//achurkin//Documents//MEGA//Imperial College London//Pierre Pinson//ViPES2X//Roar Nicolaisen//DK1 Day-ahead Prices_202301010000-202401010000.csv"
DA_market_data = CSV.read(DA_market_data_file_path, DataFrame)

# # Get the delivery start times from the day-ahead market data set:
DA_delivery_start_times = split.(DA_market_data[!,"MTU (CET/CEST)"], " - ")

# Parse the original string into a DateTime object
delivery_hour_to_optimise_DateTime = DateTime(delivery_hour_to_optimise, "yyyy-mm-dd HH:MM:SS")

# Convert the DateTime object into the desired format, find this delivery time in the day-ahead data, find the day-ahead price:
delivery_hour_to_optimise_time_format2 = Dates.format(delivery_hour_to_optimise_DateTime, "dd.mm.yyyy HH:MM")
DA_delivery_time_position = findall(x -> x == delivery_hour_to_optimise_time_format2, map(x -> x[1], DA_delivery_start_times))[1]
DA_delivery_time_price = DA_market_data[!,"Day-ahead Price [EUR/MWh]"][DA_delivery_time_position][1]




# # Define battery energy storage system (BESS) parameters, in MW and MWh:
BESS_ch_power_max = 1.0 # MW
BESS_dischch_power_max = 1.0 # MW
""" 
Note that charging more energy than BESS_ch_power_max is physically impossible within one delivery hour. 
Such trading actions are possible, but would imply speculative trading.
Thus, SoC in this model is virtual and session-based.
"""
BESS_energy_max = 2.0 # MWh
BESS_energy_0 = 0.0 # initial energy (state of charge), MWh
BESS_energy_cost_0 = 20 # charging cost of the initial energy capacity, EUR/MW
BESS_eta_ch = 0.90 # battery charging efficiency factor
BESS_eta_disch = 0.90 # battery discharging efficiency factor
Trading_and_Clearing_fee = 0.124 # EUR/MWh (check Nord Pool or EPEX fee schedule)




""" Building and solving the BESS trading optimisation model """

include("../src/bess_optimisation_1h.jl")

BESS_objective_value, BESS_optimisation_results = optimise_BESS_in_1_session(ID_market_data;
    E_max = BESS_energy_max, 
    P_ch_max = BESS_ch_power_max, 
    P_disch_max = BESS_dischch_power_max, 
    eta_ch = BESS_eta_ch, 
    eta_disch = BESS_eta_disch, 
    SoC_init = BESS_energy_0, 
    SoC_final = BESS_energy_0, 
    E_cost_0 = BESS_energy_cost_0,
    fee = Trading_and_Clearing_fee
)

BESS_optimisation_results_only_actions = filter(row -> (row.volume_ch != 0 || row.volume_disch != 0),
                  BESS_optimisation_results
)

println("\nBESS optimisation results, actions only:")
println(BESS_optimisation_results_only_actions)




""" Visualising the intraday prices together with the day-ahead price and the optimal BESS trading actions """

fz = 18 # fontsize

plt1 = plot(
    title = "DK1 intraday continuous market for delivery at "*delivery_hour_to_optimise
    *"\nTotal number of price changes: "*string(size(ID_market_data)[1])
    *"\u00A0\u00A0\u00A0\u00A0 Earliest order: "*string(round(minimum(ID_market_data.hours_to_delivery),digits=2))*" h"
    *"\u00A0\u00A0\u00A0\u00A0 Latest order: "*string(round(maximum(ID_market_data.hours_to_delivery),digits=2))*" h"
    *"\nDay-ahead price : "*string(DA_delivery_time_price)*" EUR/MWh"
    *"\u00A0\u00A0\u00A0\u00A0 Number of optimal BESS actions: "*string(nrow(BESS_optimisation_results_only_actions))
    *"\u00A0\u00A0\u00A0\u00A0Total profit: "*string(round(BESS_objective_value,digits=2))*" EUR",

    xlabel = "Time of bid/ask orders (hours prior to delivery)",
    ylabel = "Price, EUR/MWh",

    size = (2000,1000), # width and height of the whole plot (in px)
    # size = (2000,1500), # width and height of the whole plot (in px)

    # xlim = (-33, 0), # maximum 33 hours before the delivery time
    xlim = (minimum(hours_to_delivery), 0),
    # xlim = (-3,-1),
    # ylim = (-200,-50),

    # ylim = (DA_delivery_time_price-100, DA_delivery_time_price+100),
    # ylim = (mean(ID_market_data.bid_price) - std(ID_market_data.bid_price), mean(ID_market_data.bid_price) + std(ID_market_data.bid_price)),
    # ylim = (DA_delivery_time_price - std(ID_market_data.bid_price), DA_delivery_time_price + std(ID_market_data.bid_price)),
    ylim = (DA_delivery_time_price - std(ID_market_data.bid_price)/2, DA_delivery_time_price + std(ID_market_data.bid_price)/2),

    # ylim = (-11, 2),

    # ylim = (0, 3000),


    xtickfontsize=fz, ytickfontsize=fz,
    fontfamily = "Courier", 
    titlefontsize = fz-3,
    xguidefontsize = fz,
    yguidefontsize = fz,

    # legend = false,
    legendfont = fz-3,
    # legend = :outertop,
    legend = :topleft,
    
    framestyle = :box,

    # margin = 10mm,
    left_margin = 20mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    
    minorgrid = :true,
)

plot!(plt1,
    [-33,1],
    [DA_delivery_time_price,DA_delivery_time_price],
    label = "Day-ahead price",
    # color = :pink,
    color = :grey,
    linestyle = :dash,
    w = 4
)

plot!(plt1,
    ID_market_data.hours_to_delivery, # <-- plot as hours to delivery
    ID_market_data.ask_price, 
    label = "Best ask price",
    # markersize = 5,
    # markerstrokewidth = 0,
    alpha = 0.6,
    # color = palette(:tab10)[5]
    color = palette(:bluesreds)[1],
    # seriestype = :steppre,
    seriestype = :steppost,
    # marker = :circle,
    w = 3
)

plot!(plt1,
    ID_market_data.hours_to_delivery, # <-- plot as hours to delivery
    ID_market_data.bid_price, 
    label = "Best bid price",
    # markersize = 5,
    # markerstrokewidth = 0,
    alpha = 0.6,
    # color = palette(:tab10)[3]
    color = palette(:bluesreds)[3],
    # seriestype = :steppre,
    seriestype = :steppost,
    # marker = :circle,
    w = 3
)

if nrow(BESS_optimisation_results_only_actions) != 0
    BESS_optimisation_results_only_charging = filter(:action_ch => ==(1), BESS_optimisation_results_only_actions)
    BESS_optimisation_results_only_discharging = filter(:action_disch => ==(1), BESS_optimisation_results_only_actions)

    charging_marker_sizes = 14 .* BESS_optimisation_results_only_charging.volume_ch
    discharging_marker_sizes = 14 .* BESS_optimisation_results_only_discharging.volume_disch


    scatter!(plt1,
        BESS_optimisation_results_only_charging.hours_to_delivery,
        BESS_optimisation_results_only_charging.ask_price, 
        label = "BESS charging actions",
        # markersize = 10,
        markersize = charging_marker_sizes,
        markerstrokewidth = 0.0,
        alpha = 0.65,
        # color = "#1b9e77",
        color = palette(:bluesreds)[1],
        # markerstrokecolor = palette(:bluesreds)[1],
        # label = false,
        # legend = false
    )

    # for charging_i = 1:nrow(BESS_optimisation_results_only_charging)
    #     annotate!(plt1,
    #         BESS_optimisation_results_only_charging.hours_to_delivery[charging_i],
    #         BESS_optimisation_results_only_charging.ask_price[charging_i],
    #         text(string.(round.(BESS_optimisation_results_only_charging.volume_ch[charging_i], digits=2)), :black, 8)
    #     )
    # end

    scatter!(plt1,
        BESS_optimisation_results_only_discharging.hours_to_delivery,
        BESS_optimisation_results_only_discharging.bid_price, 
        label = "BESS discharging actions",
        # markersize = 10,
        markersize = discharging_marker_sizes,
        markerstrokewidth = 0.0,
        alpha = 0.65,
        # color = "#d95f02",
        color = palette(:bluesreds)[3],
        # label = false,
        # legend = false
    )

    # for discharging_i = 1:nrow(BESS_optimisation_results_only_discharging)
    #     annotate!(plt1,
    #         BESS_optimisation_results_only_discharging.hours_to_delivery[discharging_i],
    #         BESS_optimisation_results_only_discharging.bid_price[discharging_i],
    #         text(string.(round.(BESS_optimisation_results_only_discharging.volume_disch[discharging_i], digits=2)), :black, 8)
    #     )
    # end

end

display(plt1)


savefig("../results/run_optimal_backtest_1h_test5.png")
# savefig("../results/run_optimal_backtest_1h_test1.svg")
# savefig("../results/run_optimal_backtest_1h_test1.pdf")


