#include "newtonpf.hpp"

#include <algorithm>
#include <vector>
#include <complex>



void Jacobian::analyze(const nr_data::YbusType &Ybus, const nr_data::VectorXi32 &pv, const nr_data::VectorXi32 &pq) {
    // 1) Dimensions
    const int32_t nbus = Ybus.rows();
    const int32_t npv = pv.size();
    const int32_t npq = pq.size();
    const int32_t npvpq = npv + npq;

    // 소영수정: pv와 pq를 합친 pvpq 벡터 생성
    this->pvpq.resize(npvpq);
    this->pq = pq;
    for (int i = 0; i < npv; ++i) this->pvpq(i) = pv(i);
    for (int i = 0; i < npq; ++i) this->pvpq(npv + i) = pq(i);

    r1 = c1 = npvpq;
    r2 = c2 = npq;
    R = C = npvpq + npq;

    // 2) Global bus -> Jacobian row/col maps
    std::vector<int> rmap_pvpq(nbus, -1);
    std::vector<int> rmap_pq  (nbus, -1);
    std::vector<int> cmap_pvpq(nbus, -1);
    std::vector<int> cmap_pq  (nbus, -1);

    for (int i = 0; i < npvpq; ++i) rmap_pvpq[this->pvpq(i)] = i;  // 소영수정: pvpq -> this->pvpq
    for (int i = 0; i < npq;   ++i) rmap_pq  [pq(i)]   = i + npvpq;
    for (int j = 0; j < npvpq; ++j) cmap_pvpq[this->pvpq(j)] = j;  // 소영수정: pvpq -> this->pvpq
    for (int j = 0; j < npq;   ++j) cmap_pq  [pq(j)]   = j + npvpq;

    // 3) Off-diagonal pattern (triplets)
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(4 * Ybus.nonZeros() + 4 * nbus);
    double dummy = 1.0;

    for (int k = 0; k < Ybus.outerSize(); ++k) {
        for (nr_data::YbusType::InnerIterator it(Ybus, k); it; ++it) {
        const int Y_i = it.row();
        const int Y_j = it.col();
        if (Y_i == Y_j) continue;

        const int J_i_pvpq = rmap_pvpq[Y_i];
        const int J_i_pq   = rmap_pq  [Y_i];
        const int J_j_pvpq = cmap_pvpq[Y_j];
        const int J_j_pq   = cmap_pq  [Y_j];

        if (J_i_pvpq >= 0 && J_j_pvpq >= 0) trips.emplace_back(J_i_pvpq, J_j_pvpq, dummy);
        if (J_i_pq   >= 0 && J_j_pvpq >= 0) trips.emplace_back(J_i_pq,   J_j_pvpq, dummy);
        if (J_i_pvpq >= 0 && J_j_pq   >= 0) trips.emplace_back(J_i_pvpq, J_j_pq,   dummy);
        if (J_i_pq   >= 0 && J_j_pq   >= 0) trips.emplace_back(J_i_pq,   J_j_pq,   dummy);
        }
    }

    // 4) Diagonal pattern
    for (int bus = 0; bus < nbus; ++bus) {
        const int J_i_pvpq = rmap_pvpq[bus];
        const int J_i_pq   = rmap_pq  [bus];
        const int J_j_pvpq = cmap_pvpq[bus];
        const int J_j_pq   = cmap_pq  [bus];

        if (J_i_pvpq >= 0 && J_j_pvpq >= 0) trips.emplace_back(J_i_pvpq, J_j_pvpq, dummy);
        if (J_i_pq   >= 0 && J_j_pvpq >= 0) trips.emplace_back(J_i_pq,   J_j_pvpq, dummy);
        if (J_i_pvpq >= 0 && J_j_pq   >= 0) trips.emplace_back(J_i_pvpq, J_j_pq,   dummy);
        if (J_i_pq   >= 0 && J_j_pq   >= 0) trips.emplace_back(J_i_pq,   J_j_pq,   dummy);
    }

    // 5) Build sparse J
    J.resize(R, C);
    J.setFromTriplets(trips.begin(), trips.end());
    J.makeCompressed();

    // 6) Helper: coefficient index finder (ColMajor)
    // Find index p in valuePtr() for (row, col)
    auto find_coeff_index = [&](int row, int col) -> int {
        const int *inner = J.innerIndexPtr(); // row indices (ColMajor)
        const int *outer = J.outerIndexPtr(); // column pointers
        const int *start = inner + outer[col];
        const int *end   = inner + outer[col + 1];
        const int *it    = std::lower_bound(start, end, row);
        if (it != end && *it == row) return static_cast<int>(it - inner);
        return -1;
    };

    // 7) Off-diagonal maps (Ybus nnz → J value index)
    const int nz = Ybus.nonZeros();
    mapJ11.assign(nz, -1);
    mapJ21.assign(nz, -1);
    mapJ12.assign(nz, -1);
    mapJ22.assign(nz, -1);

    int t = 0;
    for (int k = 0; k < Ybus.outerSize(); ++k) {
        for (nr_data::YbusType::InnerIterator it(Ybus, k); it; ++it, ++t) {
        const int Y_i = it.row();
        const int Y_j = it.col();

        const int J_i_pvpq = rmap_pvpq[Y_i];
        const int J_i_pq   = rmap_pq  [Y_i];
        const int J_j_pvpq = cmap_pvpq[Y_j];
        const int J_j_pq   = cmap_pq  [Y_j];

        if (J_i_pvpq >= 0 && J_j_pvpq >= 0) mapJ11[t] = find_coeff_index(J_i_pvpq, J_j_pvpq);
        if (J_i_pq   >= 0 && J_j_pvpq >= 0) mapJ21[t] = find_coeff_index(J_i_pq,   J_j_pvpq);
        if (J_i_pvpq >= 0 && J_j_pq   >= 0) mapJ12[t] = find_coeff_index(J_i_pvpq, J_j_pq);
        if (J_i_pq   >= 0 && J_j_pq   >= 0) mapJ22[t] = find_coeff_index(J_i_pq,   J_j_pq);
        }
    }

    // 8) Diagonal maps
    diagMapJ11.assign(nbus, -1);
    diagMapJ21.assign(nbus, -1);
    diagMapJ12.assign(nbus, -1);
    diagMapJ22.assign(nbus, -1);

    for (int bus = 0; bus < nbus; ++bus) {
        const int J_i_pvpq = rmap_pvpq[bus];
        const int J_i_pq   = rmap_pq  [bus];
        const int J_j_pvpq = cmap_pvpq[bus];
        const int J_j_pq   = cmap_pq  [bus];

        if (J_i_pvpq >= 0 && J_j_pvpq >= 0) diagMapJ11[bus] = find_coeff_index(J_i_pvpq, J_j_pvpq);
        if (J_i_pq   >= 0 && J_j_pvpq >= 0) diagMapJ21[bus] = find_coeff_index(J_i_pq,   J_j_pvpq);
        if (J_i_pvpq >= 0 && J_j_pq   >= 0) diagMapJ12[bus] = find_coeff_index(J_i_pvpq, J_j_pq);
        if (J_i_pq   >= 0 && J_j_pq   >= 0) diagMapJ22[bus] = find_coeff_index(J_i_pq,   J_j_pq);
    }
}

void Jacobian::update(const nr_data::YbusType &Ybus, const nr_data::VectorXcd &V, const nr_data::VectorXcd &Ibus) {
    using cxd = std::complex<double>;
    const int nb = Ybus.rows();

    // 1) Pointers & intermediates
    double *J_ptr = J.valuePtr();
    nr_data::VectorXd Vm  = V.cwiseAbs().cwiseMax(1e-8);
    nr_data::VectorXcd Vnorm = V.cwiseQuotient(Vm.cast<cxd>());
    nr_data::VectorXcd Sbus  = V.cwiseProduct(Ibus.conjugate());
    const cxd j(0.0, 1.0);

    // 2) Off-diagonals
    int t = 0;
    for (int k = 0; k < Ybus.outerSize(); ++k) {
        for (nr_data::YbusType::InnerIterator it(Ybus, k); it; ++it, ++t) {
        const int Y_i = it.row();
        const int Y_j = it.col();
        const cxd y   = it.value();

        const cxd va = -j * V(Y_i) * std::conj(y * V(Y_j));
        const cxd vm =  V(Y_i) * std::conj(y * Vnorm(Y_j));

        const int p11 = mapJ11[t];
        const int p21 = mapJ21[t];
        const int p12 = mapJ12[t];
        const int p22 = mapJ22[t];

        if (p11 >= 0) J_ptr[p11] = std::real(va);
        if (p21 >= 0) J_ptr[p21] = std::imag(va);
        if (p12 >= 0) J_ptr[p12] = std::real(vm);
        if (p22 >= 0) J_ptr[p22] = std::imag(vm);
        }
    }

    // 3) Diagonals
    for (int bus = 0; bus < nb; ++bus) {
        const cxd va = j * Sbus(bus);
        const cxd vm = std::conj(Ibus(bus)) * Vnorm(bus);

        const int q11 = diagMapJ11[bus];
        const int q21 = diagMapJ21[bus];
        const int q12 = diagMapJ12[bus];
        const int q22 = diagMapJ22[bus];

        if (q11 >= 0) J_ptr[q11] += std::real(va);
        if (q21 >= 0) J_ptr[q21] += std::imag(va);
        if (q12 >= 0) J_ptr[q12] += std::real(vm);
        if (q22 >= 0) J_ptr[q22] += std::imag(vm);
    }
}


