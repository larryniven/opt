#ifndef OPT_H
#define OPT_H

#include "ebt.h"

namespace opt {

    void const_step_update(ebt::SparseVector& theta,
        ebt::SparseVector const& grad,
        double step_size);

    void const_step_update(std::vector<double>& theta,
        std::vector<double> const& grad,
        double step_size);

    void const_step_update_momentum(ebt::SparseVector& theta,
        ebt::SparseVector& update,
        ebt::SparseVector const& grad,
        double momentum,
        double step_size);

    void const_step_update_momentum(std::vector<double>& theta,
        std::vector<double>& update,
        std::vector<double> const& grad,
        double momentum,
        double step_size);

    void pa_update(ebt::SparseVector& theta,
        ebt::SparseVector const& loss_grad,
        double loss);
    
    void pa_update(std::vector<double>& theta,
        std::vector<double> const& loss_grad,
        double loss);
    
    void adagrad_update(ebt::SparseVector& theta,
        ebt::SparseVector const& loss_grad,
        ebt::SparseVector& accu_grad_sq,
        double step_size);

    void adagrad_update(std::vector<double>& theta,
        std::vector<double> const& loss_grad,
        std::vector<double>& accu_grad_sq,
        double step_size);

    void adagrad_update(std::vector<std::vector<double>>& theta,
        std::vector<std::vector<double>> const& loss_grad,
        std::vector<std::vector<double>>& accu_grad_sq,
        double step_size);

}

#endif
