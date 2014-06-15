#ifndef OPT_H
#define OPT_H

#include "ebt.h"

void pa_update(ebt::SparseVector& theta,
    ebt::SparseVector const& loss_grad,
    double loss);

void adagrad_update(ebt::SparseVector& theta,
    ebt::SparseVector const& loss_grad,
    ebt::SparseVector& accu_grad_sq,
    double step_size);

#endif
