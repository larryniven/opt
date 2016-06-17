#ifndef OPT_H
#define OPT_H

#include "ebt/ebt.h"
#include "la/la.h"

namespace opt {

    void const_step_update(ebt::SparseVector& theta,
        ebt::SparseVector const& grad,
        double step_size);

    void const_step_update(std::vector<double>& theta,
        std::vector<double> const& grad,
        double step_size);

    void const_step_update(std::vector<std::vector<double>>& theta,
        std::vector<std::vector<double>> const& grad,
        double step_size);

    void const_step_update(la::vector_like<double>& theta,
        la::vector_like<double> const& grad,
        double step_size);

    void const_step_update(la::matrix_like<double>& theta,
        la::matrix_like<double> const& grad,
        double step_size);

    void const_step_update_momentum(ebt::SparseVector& theta,
        ebt::SparseVector const& grad,
        ebt::SparseVector& update,
        double momentum,
        double step_size);

    void const_step_update_momentum(std::vector<double>& theta,
        std::vector<double> const& grad,
        std::vector<double>& update,
        double momentum,
        double step_size);

    void const_step_update_momentum(std::vector<std::vector<double>>& theta,
        std::vector<std::vector<double>> const& grad,
        std::vector<std::vector<double>>& update,
        double momentum,
        double step_size);

    void const_step_update_momentum(la::vector_like<double>& theta,
        la::vector_like<double> const& grad,
        la::vector_like<double>& update,
        double momentum,
        double step_size);

    void const_step_update_momentum(la::matrix_like<double>& theta,
        la::matrix_like<double> const& grad,
        la::matrix_like<double>& update,
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

    void adagrad_update(la::vector_like<double>& theta,
        la::vector_like<double> const& loss_grad,
        la::vector_like<double>& accu_grad_sq,
        double step_size);

    void adagrad_update(la::matrix_like<double>& theta,
        la::matrix_like<double> const& loss_grad,
        la::matrix_like<double>& accu_grad_sq,
        double step_size);

    void adagrad_update(std::vector<double>& theta,
        std::vector<double> const& loss_grad,
        std::vector<double>& accu_grad_sq,
        double step_size);

    void adagrad_update(std::vector<float>& theta,
        std::vector<float> const& loss_grad,
        std::vector<float>& accu_grad_sq,
        float step_size);

    void adagrad_update(std::vector<std::vector<double>>& theta,
        std::vector<std::vector<double>> const& loss_grad,
        std::vector<std::vector<double>>& accu_grad_sq,
        double step_size);

    void adagrad_update(std::vector<std::vector<float>>& theta,
        std::vector<std::vector<float>> const& loss_grad,
        std::vector<std::vector<float>>& accu_grad_sq,
        float step_size);

    void rmsprop_update(la::vector_like<double>& theta,
        la::vector_like<double> const& loss_grad,
        la::vector_like<double>& accu_grad_sq,
        double decay,
        double step_size);

    void rmsprop_update(la::matrix_like<double>& theta,
        la::matrix_like<double> const& loss_grad,
        la::matrix_like<double>& accu_grad_sq,
        double decay,
        double step_size);

    void adam_update(la::vector_like<double>& theta,
        la::vector_like<double> const& loss_grad,
        la::vector_like<double>& first_moment,
        la::vector_like<double>& second_moment,
        double time, double alpha, double beta1, double beta2);

    void adam_update(la::matrix_like<double>& theta,
        la::matrix_like<double> const& loss_grad,
        la::matrix_like<double>& first_moment,
        la::matrix_like<double>& second_moment,
        double time, double alpha, double beta1, double beta2);

}

#endif
