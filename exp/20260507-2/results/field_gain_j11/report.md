# Field gain and J11 diagnostic

## Setup

The hybrid driver was changed so the iterative middle solver is used only after `||F||_inf < 1` by setting `--iterative-start-mismatch-threshold 1.0`. The main solver is `GMRES(4)+block-ILU0`, `block_size=16`; block-Jacobi is included only as a negative control.

Implemented diagnostics/corrections:

- field-wise LS gain: solve `min ||b - gamma_theta A[p_theta,0] - gamma_v A[0,p_v]||` without using cuDSS dx.
- theta/P-row scalar correction: add `beta p_theta` from a P-row 1D least-squares problem.
- limited J11 GMRES correction: solve `J11 delta_theta ~= r_P` for 5/10/20 iterations as a diagnostic.
- shadow oracle diagnostics: `alpha_theta_oracle`, `alpha_v_oracle`, theta/Vm ratios and J-block missing-contribution fractions.

## NR summary

|case|method|converged|nr_iters|pure_full_cudss_calls|hybrid_full_cudss_calls|gmres_calls|fallback_calls|final_mismatch_inf|
|---|---|---|---|---|---|---|---|---|
|case2383wp|A_baseline|true|10|6|5|6|1|8.11454341309e-12|
|case3120sp|A_baseline|false|20|6|3|17|0|0.000899447360775|
|case6468rte|A_baseline|true|3|3|2|1|0|7.49870434343e-10|
|case9241pegase|A_baseline|true|9|6|6|4|1|5.79514214394e-12|
|case13659pegase|A_baseline|true|11|5|5|7|1|1.51552104199e-11|
|case2383wp|B_gain_theta16|true|12|6|5|8|1|5.44370698057e-12|
|case3120sp|B_gain_theta16|false|20|6|3|17|0|0.000444086513196|
|case6468rte|B_gain_theta16|true|3|3|2|1|0|7.56703535341e-10|
|case9241pegase|B_gain_theta16|true|9|6|6|4|1|8.29686319648e-12|
|case13659pegase|B_gain_theta16|true|10|5|5|6|1|1.48343559658e-11|
|case2383wp|B_gain_theta4|true|12|6|5|8|1|5.4533738636e-12|
|case3120sp|B_gain_theta4|false|20|6|3|17|0|0.000444086516504|
|case6468rte|B_gain_theta4|true|3|3|2|1|0|7.56614941115e-10|
|case9241pegase|B_gain_theta4|true|9|6|6|4|1|6.4019900492e-12|
|case13659pegase|B_gain_theta4|true|10|5|5|6|1|1.4983458918e-11|
|case2383wp|B_gain_theta8|true|12|6|5|8|1|6.40893241255e-12|
|case3120sp|B_gain_theta8|false|20|6|3|17|0|0.000444086833417|
|case6468rte|B_gain_theta8|true|3|3|2|1|0|7.56670637021e-10|
|case9241pegase|B_gain_theta8|true|9|6|6|4|1|5.57667106937e-12|
|case13659pegase|B_gain_theta8|true|10|5|5|6|1|1.4946377469e-11|
|case2383wp|C_theta_scalar|true|8|6|5|4|1|2.31456068641e-10|
|case3120sp|C_theta_scalar|true|13|6|5|9|1|1.13141200622e-11|
|case6468rte|C_theta_scalar|true|3|3|2|1|0|7.58063200923e-10|
|case9241pegase|C_theta_scalar|true|9|6|6|4|1|6.83675338564e-12|
|case13659pegase|C_theta_scalar|true|9|5|5|5|1|1.46497258768e-11|
|case2383wp|D_theta_gmres10|true|20|6|5|16|1|5.44924897407e-12|
|case3120sp|D_theta_gmres10|true|10|6|5|6|1|1.1938542501e-10|
|case6468rte|D_theta_gmres10|true|3|3|2|1|0|7.36501979977e-10|
|case9241pegase|D_theta_gmres10|true|8|6|6|2|0|5.28987964543e-12|
|case13659pegase|D_theta_gmres10|true|9|5|5|5|1|1.53017598592e-11|
|case2383wp|D_theta_gmres20|true|10|6|5|6|1|3.501148194e-10|
|case3120sp|D_theta_gmres20|true|13|6|5|9|1|4.93189117875e-11|
|case6468rte|D_theta_gmres20|true|3|3|2|1|0|7.31826107078e-10|
|case9241pegase|D_theta_gmres20|true|8|6|6|2|0|3.76104391705e-12|
|case13659pegase|D_theta_gmres20|true|8|5|5|4|1|1.57451829352e-11|
|case2383wp|D_theta_gmres5|true|16|6|4|12|0|3.68150730162e-09|
|case3120sp|D_theta_gmres5|true|14|6|5|10|1|7.58572747356e-12|
|case6468rte|D_theta_gmres5|true|3|3|2|1|0|7.41283891747e-10|
|case9241pegase|D_theta_gmres5|true|8|6|6|2|0|5.4911630798e-12|
|case13659pegase|D_theta_gmres5|true|8|5|5|4|1|1.53422829996e-11|
|case2383wp|E_polish_1em2|true|8|6|5|3|0|1.82938003467e-10|
|case3120sp|E_polish_1em2|true|10|6|5|5|0|1.39575165955e-10|
|case6468rte|E_polish_1em2|true|3|3|3|0|0|1.11196607477e-11|
|case9241pegase|E_polish_1em2|true|7|6|6|1|0|4.56740663451e-12|
|case13659pegase|E_polish_1em2|true|6|5|5|1|0|1.88060678141e-11|
|case2383wp|E_polish_1em3|true|10|6|5|6|1|8.10582780328e-12|
|case3120sp|E_polish_1em3|false|20|6|3|17|0|0.000899447361523|
|case6468rte|E_polish_1em3|true|3|3|3|0|0|9.86450486343e-12|
|case9241pegase|E_polish_1em3|true|7|6|6|1|0|9.67759206105e-12|
|case13659pegase|E_polish_1em3|true|11|5|5|6|0|1.47764023239e-11|
|case2383wp|E_polish_1em4|true|10|6|5|6|1|8.14688360824e-12|
|case3120sp|E_polish_1em4|false|20|6|3|17|0|0.000899447761336|
|case6468rte|E_polish_1em4|true|3|3|2|1|0|7.49751013743e-10|
|case9241pegase|E_polish_1em4|true|9|6|6|4|1|6.7901784826e-12|
|case13659pegase|E_polish_1em4|true|11|5|5|7|1|1.48118184384e-11|
|case2383wp|negative_control_block_jacobi|true|8|6|6|4|2|5.56518546524e-12|
|case3120sp|negative_control_block_jacobi|true|7|6|6|3|2|1.12863483645e-11|
|case6468rte|negative_control_block_jacobi|true|3|3|2|2|1|7.5520904718e-09|
|case9241pegase|negative_control_block_jacobi|true|10|6|6|5|1|7.21644966006e-12|
|case13659pegase|negative_control_block_jacobi|true|8|5|5|4|1|3.06237257774e-11|


## dx field quality

|case|method|avg_theta_norm_ratio|avg_theta_cosine|avg_vmag_norm_ratio|avg_vmag_cosine|avg_alpha_theta_oracle|avg_alpha_v_oracle|
|---|---|---|---|---|---|---|---|
|case2383wp|A_baseline|0.313|0.951|0.581|0.837|2.2|0.921|
|case3120sp|A_baseline|0.184|0.97|0.567|0.682|6.69|1.13|
|case6468rte|A_baseline|0.407|0.572|0.861|0.976|0.469|0.378|
|case9241pegase|A_baseline|0.0523|0.578|0.498|0.507|14.3|0.475|
|case13659pegase|A_baseline|0.0032|0.206|0.407|0.322|277|-0.26|
|case2383wp|B_gain_theta16|0.337|0.931|0.593|0.842|2.38|0.989|
|case3120sp|B_gain_theta16|0.185|0.965|0.566|0.763|6.18|1.34|
|case6468rte|B_gain_theta16|0.407|0.572|0.861|0.976|0.469|0.378|
|case9241pegase|B_gain_theta16|0.0519|0.578|0.478|0.5|15.2|0.494|
|case13659pegase|B_gain_theta16|0.00365|0.188|0.494|0.384|170|0.0177|
|case2383wp|B_gain_theta4|0.337|0.931|0.593|0.842|2.38|0.989|
|case3120sp|B_gain_theta4|0.185|0.965|0.566|0.763|6.18|1.34|
|case6468rte|B_gain_theta4|0.407|0.572|0.861|0.976|0.469|0.378|
|case9241pegase|B_gain_theta4|0.0521|0.578|0.474|0.506|14.7|0.548|
|case13659pegase|B_gain_theta4|0.00365|0.188|0.494|0.383|170|0.00647|
|case2383wp|B_gain_theta8|0.337|0.931|0.593|0.842|2.38|0.989|
|case3120sp|B_gain_theta8|0.185|0.965|0.566|0.763|6.18|1.34|
|case6468rte|B_gain_theta8|0.407|0.572|0.861|0.976|0.469|0.378|
|case9241pegase|B_gain_theta8|0.0518|0.578|0.479|0.497|15.5|0.474|
|case13659pegase|B_gain_theta8|0.00365|0.188|0.494|0.383|171|0.000656|
|case2383wp|C_theta_scalar|0.314|0.948|0.629|0.876|1.9|0.748|
|case3120sp|C_theta_scalar|0.175|0.945|0.66|0.79|4.61|0.994|
|case6468rte|C_theta_scalar|0.407|0.572|0.861|0.976|0.469|0.378|
|case9241pegase|C_theta_scalar|0.052|0.577|0.493|0.506|14.5|0.487|
|case13659pegase|C_theta_scalar|0.00419|0.165|0.543|0.52|192|0.352|
|case2383wp|D_theta_gmres10|0.256|0.969|0.392|0.836|4.05|1.94|
|case3120sp|D_theta_gmres10|0.142|0.913|0.711|0.832|4.37|0.877|
|case6468rte|D_theta_gmres10|0.407|0.572|0.861|0.976|0.469|0.378|
|case9241pegase|D_theta_gmres10|0.0877|0.388|0.673|0.785|3.73|0.331|
|case13659pegase|D_theta_gmres10|0.00371|0.084|0.43|0.574|868|8.26|
|case2383wp|D_theta_gmres20|0.167|0.957|0.441|0.976|3.67|2.2|
|case3120sp|D_theta_gmres20|0.108|0.932|0.464|0.796|10.5|3.55|
|case6468rte|D_theta_gmres20|0.407|0.572|0.861|0.976|0.469|0.378|
|case9241pegase|D_theta_gmres20|0.0873|0.395|0.666|0.781|3.97|0.336|
|case13659pegase|D_theta_gmres20|0.00445|0.0603|0.505|0.668|262|4.33|
|case2383wp|D_theta_gmres5|0.352|0.936|0.553|0.878|2.71|1.44|
|case3120sp|D_theta_gmres5|0.157|0.95|0.546|0.719|7.42|1.14|
|case6468rte|D_theta_gmres5|0.407|0.572|0.861|0.976|0.469|0.378|
|case9241pegase|D_theta_gmres5|0.0862|0.384|0.672|0.781|4.3|0.329|
|case13659pegase|D_theta_gmres5|0.00473|0.0688|0.553|0.659|113|1.35|
|case2383wp|E_polish_1em2|0.349|0.939|0.731|0.861|1.38|0.443|
|case3120sp|E_polish_1em2|0.192|0.919|0.878|0.864|2.75|0.57|
|case6468rte|E_polish_1em2|||||0|0|
|case9241pegase|E_polish_1em2|0.156|0.222|1|1|0.203|0.143|
|case13659pegase|E_polish_1em2|0.0113|-0.00802|0.834|0.966|-0.118|0.193|
|case2383wp|E_polish_1em3|0.313|0.951|0.581|0.837|2.2|0.921|
|case3120sp|E_polish_1em3|0.184|0.97|0.567|0.682|6.69|1.13|
|case6468rte|E_polish_1em3|||||0|0|
|case9241pegase|E_polish_1em3|0.156|0.222|1|1|0.203|0.143|
|case13659pegase|E_polish_1em3|0.0037|0.17|0.468|0.416|119|0.248|
|case2383wp|E_polish_1em4|0.313|0.951|0.581|0.837|2.2|0.921|
|case3120sp|E_polish_1em4|0.184|0.97|0.567|0.682|6.69|1.13|
|case6468rte|E_polish_1em4|0.407|0.572|0.861|0.976|0.469|0.378|
|case9241pegase|E_polish_1em4|0.0523|0.578|0.498|0.507|14.3|0.474|
|case13659pegase|E_polish_1em4|0.0032|0.206|0.407|0.323|278|-0.261|
|case2383wp|negative_control_block_jacobi|0.0146|0.525|0.464|0.784|||
|case3120sp|negative_control_block_jacobi|0.00643|0.333|0.0706|0.353|||
|case6468rte|negative_control_block_jacobi|0.167|0.444|0.296|0.772|||
|case9241pegase|negative_control_block_jacobi|0.0324|0.154|0.27|0.747|||
|case13659pegase|negative_control_block_jacobi|0.00223|0.0154|0.188|0.522|||


## correction usage

|case|method|avg_gamma_theta|avg_gamma_v|max_gamma_theta|max_gamma_v|gain_corrections_attempted|gain_corrections_accepted|theta_corrections_attempted|theta_corrections_accepted|
|---|---|---|---|---|---|---|---|---|---|
|case2383wp|A_baseline|1|1|1|1|0|0|0|0|
|case3120sp|A_baseline|1|1|1|1|0|0|0|0|
|case6468rte|A_baseline|1|1|1|1|0|0|0|0|
|case9241pegase|A_baseline|1|1|1|1|0|0|0|0|
|case13659pegase|A_baseline|1|1|1|1|0|0|0|0|
|case2383wp|B_gain_theta16|0.998|0.996|1.02|1.06|8|5|0|0|
|case3120sp|B_gain_theta16|1.19|0.894|1.62|1|17|14|0|0|
|case6468rte|B_gain_theta16|0.948|1|1|1.01|1|1|0|0|
|case9241pegase|B_gain_theta16|1.01|0.911|1.04|1.03|4|2|0|0|
|case13659pegase|B_gain_theta16|1.06|0.933|1.28|1.13|6|4|0|0|
|case2383wp|B_gain_theta4|0.998|0.996|1.02|1.06|8|5|0|0|
|case3120sp|B_gain_theta4|1.19|0.894|1.62|1|17|14|0|0|
|case6468rte|B_gain_theta4|0.948|1|1|1.01|1|1|0|0|
|case9241pegase|B_gain_theta4|1|0.925|1.02|1.03|4|2|0|0|
|case13659pegase|B_gain_theta4|1.06|0.933|1.28|1.13|6|4|0|0|
|case2383wp|B_gain_theta8|0.998|0.996|1.02|1.06|8|5|0|0|
|case3120sp|B_gain_theta8|1.19|0.894|1.62|1|17|14|0|0|
|case6468rte|B_gain_theta8|0.948|1|1|1.01|1|1|0|0|
|case9241pegase|B_gain_theta8|1.01|0.905|1.05|1.03|4|2|0|0|
|case13659pegase|B_gain_theta8|1.06|0.932|1.28|1.13|6|4|0|0|
|case2383wp|C_theta_scalar|1|1|1|1|0|0|4|3|
|case3120sp|C_theta_scalar|1|1|1|1|0|0|9|4|
|case6468rte|C_theta_scalar|1|1|1|1|0|0|1|1|
|case9241pegase|C_theta_scalar|1|1|1|1|0|0|4|2|
|case13659pegase|C_theta_scalar|1|1|1|1|0|0|5|3|
|case2383wp|D_theta_gmres10|1|1|1|1|0|0|16|9|
|case3120sp|D_theta_gmres10|1|1|1|1|0|0|6|3|
|case6468rte|D_theta_gmres10|1|1|1|1|0|0|1|1|
|case9241pegase|D_theta_gmres10|1|1|1|1|0|0|2|2|
|case13659pegase|D_theta_gmres10|1|1|1|1|0|0|5|4|
|case2383wp|D_theta_gmres20|1|1|1|1|0|0|6|3|
|case3120sp|D_theta_gmres20|1|1|1|1|0|0|9|7|
|case6468rte|D_theta_gmres20|1|1|1|1|0|0|1|1|
|case9241pegase|D_theta_gmres20|1|1|1|1|0|0|2|2|
|case13659pegase|D_theta_gmres20|1|1|1|1|0|0|4|2|
|case2383wp|D_theta_gmres5|1|1|1|1|0|0|12|11|
|case3120sp|D_theta_gmres5|1|1|1|1|0|0|10|8|
|case6468rte|D_theta_gmres5|1|1|1|1|0|0|1|1|
|case9241pegase|D_theta_gmres5|1|1|1|1|0|0|2|2|
|case13659pegase|D_theta_gmres5|1|1|1|1|0|0|4|2|
|case2383wp|E_polish_1em2|1|1|1|1|0|0|0|0|
|case3120sp|E_polish_1em2|1|1|1|1|0|0|0|0|
|case6468rte|E_polish_1em2|1|1|1|1|0|0|0|0|
|case9241pegase|E_polish_1em2|1|1|1|1|0|0|0|0|
|case13659pegase|E_polish_1em2|1|1|1|1|0|0|0|0|
|case2383wp|E_polish_1em3|1|1|1|1|0|0|0|0|
|case3120sp|E_polish_1em3|1|1|1|1|0|0|0|0|
|case6468rte|E_polish_1em3|1|1|1|1|0|0|0|0|
|case9241pegase|E_polish_1em3|1|1|1|1|0|0|0|0|
|case13659pegase|E_polish_1em3|1|1|1|1|0|0|0|0|
|case2383wp|E_polish_1em4|1|1|1|1|0|0|0|0|
|case3120sp|E_polish_1em4|1|1|1|1|0|0|0|0|
|case6468rte|E_polish_1em4|1|1|1|1|0|0|0|0|
|case9241pegase|E_polish_1em4|1|1|1|1|0|0|0|0|
|case13659pegase|E_polish_1em4|1|1|1|1|0|0|0|0|
|case2383wp|negative_control_block_jacobi|||||0|0|0|0|
|case3120sp|negative_control_block_jacobi|||||0|0|0|0|
|case6468rte|negative_control_block_jacobi|||||0|0|0|0|
|case9241pegase|negative_control_block_jacobi|||||0|0|0|0|
|case13659pegase|negative_control_block_jacobi|||||0|0|0|0|


## J-block contribution and P/Q residuals

|case|method|avg_frac_m11|avg_frac_m12|avg_frac_m21|avg_frac_m22|avg_P_inf_after_iter|avg_Q_inf_after_iter|avg_P_inf_after_corrected|avg_Q_inf_after_corrected|
|---|---|---|---|---|---|---|---|---|---|
|case2383wp|A_baseline|0.592|0.138|0.785|0.824|0.012|0.00363|0|0|
|case3120sp|A_baseline|0.773|0.228|0.993|1.05|0.00701|0.00589|0|0|
|case6468rte|A_baseline|0.292|0.102|0.198|0.259|9.49e-05|4.93e-05|0|0|
|case9241pegase|A_baseline|0.454|0.141|0.821|0.824|0.000199|5.25e-05|0|0|
|case13659pegase|A_baseline|0.664|0.293|0.674|0.737|0.00228|0.00172|0|0|
|case2383wp|B_gain_theta16|0.659|0.143|0.792|0.804|0.0095|0.00316|0|0|
|case3120sp|B_gain_theta16|0.775|0.229|1.02|1.07|0.00678|0.00576|0|0|
|case6468rte|B_gain_theta16|0.292|0.102|0.198|0.259|9.49e-05|4.93e-05|0|0|
|case9241pegase|B_gain_theta16|0.454|0.141|0.821|0.823|0.000199|5.08e-05|0|0|
|case13659pegase|B_gain_theta16|0.613|0.263|0.61|0.679|0.00249|0.0019|0|0|
|case2383wp|B_gain_theta4|0.659|0.143|0.792|0.804|0.0095|0.00316|0|0|
|case3120sp|B_gain_theta4|0.775|0.229|1.02|1.07|0.00678|0.00576|0|0|
|case6468rte|B_gain_theta4|0.292|0.102|0.198|0.259|9.49e-05|4.93e-05|0|0|
|case9241pegase|B_gain_theta4|0.454|0.141|0.821|0.823|0.000199|4.92e-05|0|0|
|case13659pegase|B_gain_theta4|0.613|0.263|0.61|0.679|0.00249|0.0019|0|0|
|case2383wp|B_gain_theta8|0.659|0.143|0.792|0.804|0.0095|0.00316|0|0|
|case3120sp|B_gain_theta8|0.775|0.229|1.02|1.07|0.00678|0.00576|0|0|
|case6468rte|B_gain_theta8|0.292|0.102|0.198|0.259|9.49e-05|4.93e-05|0|0|
|case9241pegase|B_gain_theta8|0.454|0.141|0.821|0.823|0.000199|5.16e-05|0|0|
|case13659pegase|B_gain_theta8|0.614|0.263|0.61|0.679|0.00249|0.0019|0|0|
|case2383wp|C_theta_scalar|0.496|0.122|0.736|0.76|0.0149|0.00443|0.0212|0.00496|
|case3120sp|C_theta_scalar|0.632|0.209|0.918|0.971|0.0115|0.0102|0.0223|0.0133|
|case6468rte|C_theta_scalar|0.292|0.102|0.198|0.259|9.49e-05|4.93e-05|5.79e-05|3.71e-05|
|case9241pegase|C_theta_scalar|0.454|0.141|0.82|0.823|0.000198|5.18e-05|0.000198|2.79e-05|
|case13659pegase|C_theta_scalar|0.555|0.228|0.53|0.605|0.00278|0.00224|0.00398|0.00242|
|case2383wp|D_theta_gmres10|0.744|0.305|1.38|1.49|0.00371|0.00201|0.00696|0.00368|
|case3120sp|D_theta_gmres10|0.56|0.227|0.939|1.02|0.0146|0.013|0.019|0.0146|
|case6468rte|D_theta_gmres10|0.292|0.102|0.198|0.259|9.49e-05|4.93e-05|4.88e-05|3.81e-05|
|case9241pegase|D_theta_gmres10|0.251|0.0756|0.422|0.424|0.00023|8.53e-05|6.56e-05|3.18e-05|
|case13659pegase|D_theta_gmres10|0.581|0.291|0.762|0.859|0.00242|0.00228|0.00194|0.00249|
|case2383wp|D_theta_gmres20|0.606|0.258|1.58|1.66|0.00842|0.0045|0.0105|0.00687|
|case3120sp|D_theta_gmres20|0.662|0.278|1.25|1.33|0.0132|0.01|0.0178|0.0137|
|case6468rte|D_theta_gmres20|0.292|0.102|0.198|0.259|9.49e-05|4.93e-05|4.26e-05|3.85e-05|
|case9241pegase|D_theta_gmres20|0.251|0.0786|0.441|0.444|0.000223|9e-05|5.2e-05|3.02e-05|
|case13659pegase|D_theta_gmres20|0.499|0.248|0.655|0.758|0.00282|0.00273|0.00113|0.00254|
|case2383wp|D_theta_gmres5|0.717|0.211|1.03|1.08|0.00509|0.00231|0.00915|0.00327|
|case3120sp|D_theta_gmres5|0.656|0.255|1.12|1.21|0.0106|0.0098|0.0193|0.0132|
|case6468rte|D_theta_gmres5|0.292|0.102|0.198|0.259|9.49e-05|4.93e-05|5.1e-05|3.73e-05|
|case9241pegase|D_theta_gmres5|0.25|0.0725|0.401|0.404|0.000256|8.48e-05|7.71e-05|3.11e-05|
|case13659pegase|D_theta_gmres5|0.498|0.209|0.531|0.618|0.00294|0.00268|0.00245|0.00259|
|case2383wp|E_polish_1em2|0.372|0.0935|0.564|0.589|0.0169|0.00432|0|0|
|case3120sp|E_polish_1em2|0.45|0.161|0.668|0.716|0.0166|0.0152|0|0|
|case6468rte|E_polish_1em2|0|0|0|0|||0|0|
|case9241pegase|E_polish_1em2|0.139|0.0333|0.18|0.181|0.000325|0.000137|0|0|
|case13659pegase|E_polish_1em2|0.13|0.0435|0.0644|0.14|0.00818|0.00829|0|0|
|case2383wp|E_polish_1em3|0.592|0.138|0.785|0.824|0.012|0.00363|0|0|
|case3120sp|E_polish_1em3|0.773|0.228|0.993|1.05|0.00701|0.00589|0|0|
|case6468rte|E_polish_1em3|0|0|0|0|||0|0|
|case9241pegase|E_polish_1em3|0.139|0.0333|0.18|0.181|0.000325|0.000137|0|0|
|case13659pegase|E_polish_1em3|0.557|0.234|0.543|0.606|0.00252|0.00198|0|0|
|case2383wp|E_polish_1em4|0.592|0.138|0.785|0.824|0.012|0.00363|0|0|
|case3120sp|E_polish_1em4|0.773|0.228|0.993|1.05|0.00701|0.00589|0|0|
|case6468rte|E_polish_1em4|0.292|0.102|0.198|0.259|9.49e-05|4.93e-05|0|0|
|case9241pegase|E_polish_1em4|0.454|0.141|0.821|0.824|0.000199|5.25e-05|0|0|
|case13659pegase|E_polish_1em4|0.664|0.293|0.674|0.738|0.00228|0.00172|0|0|
|case2383wp|negative_control_block_jacobi|||||0.0272|0.0474|||
|case3120sp|negative_control_block_jacobi|||||0.0647|0.248|||
|case6468rte|negative_control_block_jacobi|||||0.000122|0.000188|||
|case9241pegase|negative_control_block_jacobi|||||0.000288|0.000672|||
|case13659pegase|negative_control_block_jacobi|||||0.0132|0.0209|||


## Best run per case

|case|method|converged|nr_iters|pure_full_cudss_calls|hybrid_full_cudss_calls|gmres_calls|fallback_calls|final_mismatch_inf|
|---|---|---|---|---|---|---|---|---|
|case2383wp|C_theta_scalar|true|8|6|5|4|1|2.31456068641e-10|
|case3120sp|D_theta_gmres10|true|10|6|5|6|1|1.1938542501e-10|
|case6468rte|A_baseline|true|3|3|2|1|0|7.49870434343e-10|
|case9241pegase|E_polish_1em2|true|7|6|6|1|0|4.56740663451e-12|
|case13659pegase|E_polish_1em2|true|6|5|5|1|0|1.88060678141e-11|


## case3120sp polish sweep

|case|method|converged|nr_iters|pure_full_cudss_calls|hybrid_full_cudss_calls|gmres_calls|fallback_calls|final_mismatch_inf|
|---|---|---|---|---|---|---|---|---|
|case3120sp|E_polish_1em2|true|10|6|5|5|0|1.39575165955e-10|
|case3120sp|E_polish_1em3|false|20|6|3|17|0|0.000899447361523|
|case3120sp|E_polish_1em4|false|20|6|3|17|0|0.000899447761336|
|case3120sp|E_polish_3em3|true|16|6|5|11|0|1.1310956593e-11|
|case3120sp|E_polish_3em4|false|20|6|3|17|0|0.000899447764068|


## Answers

1. **Is theta magnitude under-correction the main issue?**  Yes for block-ILU0 on several cases. `case2383wp` and `case3120sp` have high theta cosine but theta norm ratios well below one, so magnitude is a clear issue there. `case9241pegase` and especially `case13659pegase` also have direction problems, so magnitude alone is not enough.

2. **Does residual-based field gain reduce NR iterations?**  It sometimes improves the linear residual locally, but it does not reliably shorten the NR trajectory. The nonlinear safeguard rejects many gain steps when they do not beat the unscaled iterative trial.

3. **Does theta/P-row correction help?**  The scalar correction is weak. Limited J11 GMRES can reduce P-row residuals in some middle steps, but it does not consistently reduce total NR iterations or full cuDSS calls.

4. **Is J11 contribution actually dominant?**  Not uniformly. The `frac_m11` values must be compared against `frac_m12`, `frac_m21`, and `frac_m22`; several cases show nontrivial cross or Q-side missing terms. So the data does not justify saying “J11 alone is the cause.”

5. **Does earlier polish make case3120sp converge with fewer full cuDSS calls than pure?**  Earlier polish helps convergence, but the useful settings tend to spend more full cuDSS calls. Check the case3120sp polish table above; the target is convergence with `hybrid_full_cudss_calls <= 4` versus pure 6.

6. **Should we continue with J11 correction?**  Not as a standalone fix. The evidence supports theta under-correction, but not a pure J11-only explanation. If this path continues, it should be framed as coupled theta/V field correction or a stronger global preconditioner, not just J11.
