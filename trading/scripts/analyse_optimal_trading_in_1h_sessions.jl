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




bess_trading_summary_per_1h_session = CSV.read("..//results//bess_trading_summary_per_1h_session_8737_results.csv", DataFrame)


histogram(
    bess_trading_summary_per_1h_session.profit,
    bins = 100,
    xlabel = "Profit, EUR",
    ylabel = "Frequency",
    title = "Distribution of BESS trading profit (in 1h sessions)",
    legend = false,
    titlefont = 8,
    fontfamily = "Courier"
)

histogram(
    bess_trading_summary_per_1h_session.number_of_actions,
    bins = 100,
    xlabel = "Number of actions",
    ylabel = "Frequency",
    title = "Total number of BESS trading actions (in 1h sessions)",
    legend = false,
    titlefont = 8,
    fontfamily = "Courier",
    color = palette(:tab10)[5]
)

fz = 16

plt_scatter = plot(
    xlabel = "Total number of optimal BESS actions",
    ylabel = "Profit per session, EUR",
    size = (1100,1000), # width and height of the whole plot (in px)
    xlim = (-20, 1000),
    ylim = (-200, 10000),
    xtickfontsize=fz, ytickfontsize=fz,
    fontfamily = "Courier", 
    titlefontsize = fz-3,
    xguidefontsize = fz,
    yguidefontsize = fz,
    legendfont = fz-3,
    legend = :topleft,
    framestyle = :box,
    # margin = 10mm,
    left_margin = 20mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    minorgrid = :true
)

scatter!(plt_scatter,
    bess_trading_summary_per_1h_session.number_of_actions,
    bess_trading_summary_per_1h_session.profit,
    xlabel = "Number of actions",
    ylabel = "Profit, EUR",
    # title = "Profit vs number of actions (in 1h Sessions)",
    legend = false,
    titlefont = 8,
    fontfamily = "Courier",
    # color = :grey,
    color = palette(:tab10)[5],
    alpha = 0.3,
    markerstrokewidth = 0,
    markersize = 6
)

display(plt_scatter)

action_profit_correlation = cor(
    bess_trading_summary_per_1h_session.number_of_actions,
    bess_trading_summary_per_1h_session.profit
)

println("Action-profit correlation: ", round(action_profit_correlation, digits=3))

savefig("..//results//bess_trading_summary_test1.png")




# filtered_sessions = filter(row ->
#     250 < row.number_of_actions < 500 &&
#     500 < row.profit < 8000,
#     bess_trading_summary_per_1h_session
# )