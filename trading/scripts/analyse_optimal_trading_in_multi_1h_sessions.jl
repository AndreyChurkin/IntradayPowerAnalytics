"""
Visualisation of optimal battery energy storage system (BESS) trading actions across multiple 1-hour intraday trading sessions.
This is a preliminary visualisation of aggregated action summaries, which will be improved later.

Andrey Churkin
https://andreychurkin.ru/

"""


using CSV, DataFrames
using Statistics
using StatsPlots, Plots.PlotMeasures

cd(dirname(@__FILE__))
println(pwd())




bess_trading_summary_per_1h_session = CSV.read("..//results//bess_optimal_backtest_summary_multi_1h_sessions_8737_results.csv", DataFrame)

fz = 16

plt_histogram_profit = histogram(
    bess_trading_summary_per_1h_session.profit,
    bins = 100,
    size = (1000,600), # width and height of the whole plot (in px)
    xlabel = "Profit, EUR", 
    ylabel = "Frequency",
    xformatter = :plain,  # disables scientific notation   
    title = "Distribution of BESS trading profit (in 1h sessions)",
    # xlim = (0, maximum(bess_trading_summary_per_1h_session.profit)),
    # ylim = (0, 2000),
    legend = false,
    fontfamily = "Courier",
    titlefontsize = fz-3,
    xtickfontsize = fz,
    ytickfontsize = fz,
    xguidefontsize = fz,
    yguidefontsize = fz,
    legendfont = fz-3,
    color = :grey,
    framestyle = :box,
    # margin = 10mm,
    left_margin = 10mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    # minorgrid = :true
)
display(plt_histogram_profit)
savefig("../results/bess_optimal_backtest_multi_1h_sessions_histogram_profit.png")
# savefig("../results/bess_optimal_backtest_multi_1h_sessions_histogram_profit.svg")
# savefig("../results/bess_optimal_backtest_multi_1h_sessions_histogram_profit.pdf")

plt_histogram_actions = histogram(
    bess_trading_summary_per_1h_session.number_of_actions,
    bins = 100,
    size = (1000,600), # width and height of the whole plot (in px)
    xlabel = "Number of actions",
    ylabel = "Frequency",
    xformatter = :plain,  # disables scientific notation   
    title = "Total number of BESS trading actions (in 1h sessions)",
    # xlim = (0, maximum(bess_trading_summary_per_1h_session.number_of_actions)),
    legend = false,
    fontfamily = "Courier",
    titlefontsize = fz-3,
    xtickfontsize = fz,
    ytickfontsize = fz,
    xguidefontsize = fz,
    yguidefontsize = fz,
    legendfont = fz-3,
    # color = palette(:tab10)[5]
    color = :grey,
    framestyle = :box,
    # margin = 10mm,
    left_margin = 10mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    # minorgrid = :true
)
display(plt_histogram_actions)
savefig("../results/bess_optimal_backtest_multi_1h_sessions_histogram_actions.png")
# savefig("../results/bess_optimal_backtest_multi_1h_sessions_histogram_actions.svg")
# savefig("../results/bess_optimal_backtest_multi_1h_sessions_histogram_actions.pdf")

plt_scatter_actions_vs_profit = plot(
    xlabel = "Total number of optimal BESS actions",
    ylabel = "Profit per session, EUR",
    size = (1100,1000), # width and height of the whole plot (in px)
    # xlim = (-20, 1000),
    # ylim = (-200, 10000),
    xtickfontsize=fz, ytickfontsize=fz,
    fontfamily = "Courier", 
    titlefontsize = fz-3,
    xguidefontsize = fz,
    yguidefontsize = fz,
    legendfont = fz-3,
    legend = :topleft,
    framestyle = :box,
    # margin = 10mm,
    left_margin = 10mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    minorgrid = :true
)

scatter!(plt_scatter_actions_vs_profit,
    bess_trading_summary_per_1h_session.number_of_actions,
    bess_trading_summary_per_1h_session.profit,
    xlabel = "Number of actions",
    ylabel = "Profit, EUR",
    xformatter = :plain,  # disables scientific notation   
    yformatter = :plain,  # disables scientific notation    
    # title = "Profit vs number of actions (in 1h Sessions)",
    legend = false,
    titlefont = 8,
    fontfamily = "Courier",
    color = :black,
    # color = palette(:tab10)[5],
    alpha = 0.2,
    markerstrokewidth = 0,
    markersize = 6
)

display(plt_scatter_actions_vs_profit)
savefig("../results/bess_optimal_backtest_multi_1h_sessions_scatter_actions_vs_profit.png")
# savefig("../results/bess_optimal_backtest_multi_1h_sessions_scatter_actions_vs_profit.svg")
# savefig("../results/bess_optimal_backtest_multi_1h_sessions_scatter_actions_vs_profit.pdf")


action_profit_correlation = cor(
    bess_trading_summary_per_1h_session.number_of_actions,
    bess_trading_summary_per_1h_session.profit
)

println("Action-profit correlation: ", round(action_profit_correlation, digits=3))




# filtered_sessions = filter(row ->
#     250 < row.number_of_actions < 500 &&
#     500 < row.profit < 8000,
#     bess_trading_summary_per_1h_session
# )