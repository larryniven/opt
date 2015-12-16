#include "opt/opt.h"

namespace opt {

    void const_step_update(ebt::SparseVector& theta,
        ebt::SparseVector const& grad,
        double step_size)
    {
        for (auto& p: grad) {
            theta(p.first) -= p.second * step_size;
        }
    }

    void const_step_update(std::vector<double>& theta,
        std::vector<double> const& grad,
        double step_size)
    {
        for (int i = 0; i < theta.size(); ++i) {
            theta.at(i) -= grad.at(i) * step_size;
        }
    }

    void const_step_update(std::vector<std::vector<double>>& theta,
        std::vector<std::vector<double>> const& grad,
        double step_size)
    {
        for (int i = 0; i < theta.size(); ++i) {
            const_step_update(theta.at(i), grad.at(i), step_size);
        }
    }

    void const_step_update_momentum(ebt::SparseVector& theta,
        ebt::SparseVector const& grad,
        ebt::SparseVector& update,
        double momentum,
        double step_size)
    {
        for (auto& p: update) {
            p.second *= momentum;
        }

        for (auto& p: grad) {
            update(p.first) += p.second * (1 - momentum);
        }

        for (auto& p: update) {
            theta(p.first) -= p.second * step_size;
        }
    }

    void const_step_update_momentum(std::vector<double>& theta,
        std::vector<double> const& grad,
        std::vector<double>& update,
        double momentum,
        double step_size)
    {
        for (int i = 0; i < update.size(); ++i) {
            update.at(i) *= momentum;
        }

        for (int i = 0; i < update.size(); ++i) {
            update.at(i) += grad.at(i) * (1 - momentum);
        }

        for (int i = 0; i < theta.size(); ++i) {
            theta.at(i) -= update.at(i) * step_size;
        }
    }

    void const_step_update_momentum(std::vector<std::vector<double>>& theta,
        std::vector<std::vector<double>> const& grad,
        std::vector<std::vector<double>>& update,
        double momentum,
        double step_size)
    {
        for (int i = 0; i < theta.size(); ++i) {
            const_step_update_momentum(theta.at(i), grad.at(i), update.at(i), momentum, step_size);
        }
    }

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
    
    void pa_update(std::vector<double>& theta,
        std::vector<double> const& loss_grad,
        double loss)
    {
        if (loss > 0) {
            double grad_norm_sq = 0;
    
            for (auto& v: loss_grad) {
                grad_norm_sq += v * v;
            }
    
            double step_size = loss / grad_norm_sq;

            std::cout << "[pa] step_size: " << step_size << std::endl;
    
            for (int i = 0; i < theta.size(); ++i) {
                theta.at(i) -= loss_grad.at(i) * step_size;
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
            if (accu_grad_sq(p.first) > 0) {
                theta(p.first) -= step_size
                    / std::sqrt(accu_grad_sq(p.first)) * p.second;
            }
        }
    }

    void adagrad_update(std::vector<double>& theta,
        std::vector<double> const& loss_grad,
        std::vector<double>& accu_grad_sq,
        double step_size)
    {
        int size = loss_grad.size();
        double* accu_grad_sq_data = accu_grad_sq.data();
        double const* loss_grad_data = loss_grad.data();
        double* theta_data = theta.data();

        for (int i = 0; i < size; ++i) {
            accu_grad_sq_data[i] += std::pow(loss_grad_data[i], 2);
        }
    
        for (int i = 0; i < size; ++i) {
            if (accu_grad_sq_data[i] > 0) {
                theta_data[i] -= loss_grad_data[i] * step_size
                    / std::sqrt(accu_grad_sq_data[i]);
            }
        }
    }

    void adagrad_update(la::vector<double>& theta,
        la::vector<double> const& loss_grad,
        la::vector<double>& accu_grad_sq,
        double step_size)
    {
        int size = loss_grad.size();
        double* accu_grad_sq_data = accu_grad_sq.data();
        double const* loss_grad_data = loss_grad.data();
        double* theta_data = theta.data();

        for (int i = 0; i < size; ++i) {
            accu_grad_sq_data[i] += std::pow(loss_grad_data[i], 2);
        }
    
        for (int i = 0; i < size; ++i) {
            if (accu_grad_sq_data[i] > 0) {
                theta_data[i] -= loss_grad_data[i] * step_size
                    / std::sqrt(accu_grad_sq_data[i]);
            }
        }
    }

    void adagrad_update(std::vector<float>& theta,
        std::vector<float> const& loss_grad,
        std::vector<float>& accu_grad_sq,
        float step_size)
    {
        int size = loss_grad.size();
        float* accu_grad_sq_data = accu_grad_sq.data();
        float const* loss_grad_data = loss_grad.data();
        float* theta_data = theta.data();

        for (int i = 0; i < size; ++i) {
            accu_grad_sq_data[i] += std::pow(loss_grad_data[i], 2);
        }
    
        for (int i = 0; i < size; ++i) {
            if (accu_grad_sq_data[i] > 0) {
                theta_data[i] -= loss_grad_data[i] * step_size
                    / std::sqrt(accu_grad_sq_data[i]);
            }
        }
    }

    void adagrad_update(std::vector<std::vector<double>>& theta,
        std::vector<std::vector<double>> const& loss_grad,
        std::vector<std::vector<double>>& accu_grad_sq,
        double step_size)
    {
        for (int i = 0; i < loss_grad.size(); ++i) {
            adagrad_update(theta.at(i), loss_grad.at(i), accu_grad_sq.at(i), step_size);
        }
    }

    void adagrad_update(std::vector<std::vector<float>>& theta,
        std::vector<std::vector<float>> const& loss_grad,
        std::vector<std::vector<float>>& accu_grad_sq,
        float step_size)
    {
        for (int i = 0; i < loss_grad.size(); ++i) {
            adagrad_update(theta.at(i), loss_grad.at(i), accu_grad_sq.at(i), step_size);
        }
    }

}
