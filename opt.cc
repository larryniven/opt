#include "opt.h"

namespace opt {

    void pa_update(ebt::SparseVector& theta,
        ebt::SparseVector const& loss_grad,
        double loss)
    {
        if (loss > 0) {
            double grad_norm_sq = 0;
    
            for (auto& p: loss_grad) {
                grad_norm_sq += p.second * p.second;
            }
    
            double step_size = loss / grad_norm_sq;

            std::cout << "[pa] step_size: " << step_size << std::endl;
    
            for (auto& p: loss_grad) {
                theta(p.first) -= p.second * step_size;
            }
        }
    }
    
    void adagrad_update(ebt::SparseVector& theta,
        ebt::SparseVector const& loss_grad,
        ebt::SparseVector& accu_grad_sq,
        double step_size)
    {
        for (auto& p: loss_grad) {
            accu_grad_sq(p.first) += p.second * p.second;
        }
    
        for (auto& p: loss_grad) {
            if (accu_grad_sq(p.first) != 0) {
                theta(p.first) -= step_size
                    / std::sqrt(accu_grad_sq(p.first)) * p.second;
            }
        }
    }

}
