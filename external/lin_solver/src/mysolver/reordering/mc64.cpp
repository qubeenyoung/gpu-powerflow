#include "mysolver/reordering/mc64.hpp"

#include <cmath>
#include <limits>
#include <queue>
#include <utility>

namespace mysolver::reordering {

bool mc64_match(int n, const int* Ap, const int* Ai, const double* Ax,
                std::vector<int>& match_col)
{
    constexpr double INF = std::numeric_limits<double>::infinity();

    // Per-column-normalized cost c(j,i) = log(colmax_j / |a_ij|) >= 0 (0 for the
    // column's largest entry). A per-column constant shift does not change the
    // optimal matching, so this is equivalent to MC64's max-product objective.
    const int nnz = Ap[n];
    std::vector<double> cost(nnz, INF);
    for (int j = 0; j < n; ++j) {
        double cmax = 0.0;
        for (int p = Ap[j]; p < Ap[j + 1]; ++p) cmax = std::max(cmax, std::abs(Ax[p]));
        if (cmax <= 0.0) continue;  // empty column -> no perfect matching
        const double lcmax = std::log(cmax);
        for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
            const double a = std::abs(Ax[p]);
            if (a > 0.0) cost[p] = lcmax - std::log(a);  // >= 0
        }
    }

    std::vector<double> u(n, 0.0), v(n, 0.0);  // dual potentials: columns, rows
    std::vector<int> mate_col(n, -1), mate_row(n, -1);

    // Warm-start (Jonker-Volgenant column+row reduction + greedy assignment): set
    // potentials and pre-match the exactly-zero-reduced-cost edges, so the SSP
    // phase below only augments the few still-unmatched columns instead of all n.
    // The greedy matches only edges with reduced cost == 0 by construction (the
    // argmin edge that set v[i]), preserving the SSP invariant; the final matching
    // is still optimal (SSP re-routes as needed).
    for (int j = 0; j < n; ++j) {
        double mn = INF;
        for (int p = Ap[j]; p < Ap[j + 1]; ++p)
            if (cost[p] < mn) mn = cost[p];
        u[j] = (mn < INF) ? mn : 0.0;
    }
    std::vector<int> vcol(n, -1);  // column that set v[i] (its edge has reduced cost 0)
    std::vector<double> vmin(n, INF);
    for (int j = 0; j < n; ++j)
        for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
            if (cost[p] == INF) continue;
            const int i = Ai[p];
            const double r = cost[p] - u[j];
            if (r < vmin[i]) { vmin[i] = r; vcol[i] = j; }
        }
    for (int i = 0; i < n; ++i) v[i] = (vmin[i] < INF) ? vmin[i] : 0.0;
    for (int i = 0; i < n; ++i) {
        const int j = vcol[i];
        if (j >= 0 && mate_col[j] == -1 && mate_row[i] == -1) {
            mate_col[j] = i;
            mate_row[i] = j;
        }
    }

    using QN = std::pair<double, int>;  // (dist, row)
    std::priority_queue<QN, std::vector<QN>, std::greater<QN>> pq;

    std::vector<double> dist(n, INF);
    std::vector<int> prevc(n, -1);  // prevc[i] = column from which row i was reached
    std::vector<char> used(n, 0);   // settled rows this phase
    std::vector<int> touched;    // rows touched this phase (for cheap reset)

    for (int s = 0; s < n; ++s) {
        if (mate_col[s] != -1) continue;  // already matched by the greedy warm-start
        for (int i : touched) { dist[i] = INF; used[i] = 0; prevc[i] = -1; }
        touched.clear();
        while (!pq.empty()) pq.pop();

        auto relax_col = [&](int j, double base) {
            for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
                if (cost[p] == INF) continue;  // zero / dropped entry
                const int i = Ai[p];
                if (used[i]) continue;
                const double nd = base + (cost[p] - u[j] - v[i]);
                if (dist[i] == INF) touched.push_back(i);
                if (nd < dist[i]) {
                    dist[i] = nd;
                    prevc[i] = j;
                    pq.push({nd, i});
                }
            }
        };

        relax_col(s, 0.0);
        int aug = -1;
        double D = 0.0;
        while (!pq.empty()) {
            const auto [d, i] = pq.top();
            pq.pop();
            if (used[i] || d > dist[i]) continue;
            used[i] = 1;
            if (mate_row[i] == -1) { aug = i; D = d; break; }  // shortest augmenting path
            relax_col(mate_row[i], d);  // matched edge has reduced cost 0 -> base = d
        }
        if (aug == -1) return false;  // structurally singular

        // Dual update (keeps reduced costs >= 0 and matched edges at 0).
        u[s] += D;
        for (int i : touched)
            if (used[i]) {
                v[i] += dist[i] - D;
                if (mate_row[i] != -1) u[mate_row[i]] += D - dist[i];
            }

        // Flip the alternating path ending at the free row `aug`.
        int i = aug;
        while (i != -1) {
            const int j = prevc[i];
            const int next_i = mate_col[j];
            mate_col[j] = i;
            mate_row[i] = j;
            i = next_i;
        }
    }

    match_col = mate_col;
    return true;
}

}  // namespace mysolver::reordering
