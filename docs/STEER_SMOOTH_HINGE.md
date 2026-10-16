# Fixed smooth hinge research recipe

`zensim_mlp_train --nonneg-distance --smooth-hinge` selects the registered
one-hidden-layer S activation. The fixed tau is 1/8: h(z)=0 for z<=0,
h(z)=4z² for 0<z<1/8, and h(z)=z-1/16 otherwise. Its derivative is 0,
8z and 1 respectively, including the continuous endpoints.

The plain CPU trainer uses this activation in paired/fused forwards,
backpropagation and evaluation. Packing emits zenpredict activation 3;
the loaded-score tape and independent f64 reference use its exact derivative.
Scale-only normalization, zero hidden biases, nonpositive output weights
and the pinned output bias retain exact identity. Output-spline packing/refit
preserves the activation. Products remain under the existing derived-input owner.

Only the plain one-layer N recipe, uniform sampling and minibatch 1 are admitted.
Other heads, monotone projection, GPU training, feature transforms, pooled
objectives, NiN and K penalties are refused. Existing recipes default the flag
off and delegate to their existing activation and arithmetic owners.
The existing shared fit driver accepts the explicit recipe token `:sh1`;
it adds only `--smooth-hinge` to an otherwise identical N argv.

The runtime and baker are pinned to merged zenanalyze f0903489; checks use
that immutable dependency without a local patch. `just steer-s-final-check`
uses invented arrays and generated pixels only: endpoint/interior derivatives,
f32/f16 identity and round trips, actual pack/refit, product/spline gradients,
unsupported-mode refusals and score/feature bit parity across native, v3 and
scalar processes and padded rows. These are engineering checks; they do not
claim a scientific or runtime gate pass. The unchanged full control replay
requires separate authorization. ActivationJson's exhaustive enum extension
remains flagged for release review in zenanalyze; no release is implied.
