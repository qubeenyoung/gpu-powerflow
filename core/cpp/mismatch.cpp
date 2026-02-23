#include "newtonpf.hpp"

void mismatch(double &normF,
              nr_data::VectorXd &F,
              const nr_data::VectorXcd &V,
              const nr_data::VectorXcd &Ibus,
              const nr_data::VectorXcd &Sbus,
              const nr_data::VectorXi32 &pv,
              const nr_data::VectorXi32 &pq)
{
    // mis = V .* conj(Ibus) - Sbus
    nr_data::VectorXcd mis = V.array() * Ibus.array().conjugate() - Sbus.array();

    const uint32_t npv = pv.size();
    const uint32_t npq = pq.size();

    int k = 0;
    for (uint32_t i = 0; i < npv; i++)
        F[k++] = mis[pv[i]].real();
    for (uint32_t i = 0; i < npq; i++)
        F[k++] = mis[pq[i]].real();
    for (uint32_t i = 0; i < npq; i++)
        F[k++] = mis[pq[i]].imag();

    normF = F.cwiseAbs().maxCoeff();
}