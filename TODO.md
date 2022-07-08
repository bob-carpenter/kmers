(1) Figure out why gradients aren't matching the function values. Is it a question of scale (large function values that are unnormalized)? Or is it an actual bug in the gradient code? Why are the gradient of the BFGS model working better?

(2) right mirror_bfgs solver for simplex_model. Need only incorporate the mirror step.