# Regrade of this trial

The `verifier/` output beside this directory records **reward 0.0**, which was a
false reject. It is kept unaltered: it is what the verifier actually returned at
the time, and rewriting it would falsify the record.

    verifier/reward.json   0.0        rejected: min per-tensor cosine 0.8541 < 0.9
                                      on encoder.layer.0.attention.self.key.bias
    regrade/reward.json    0.864497   acc 0.6242, recovery 0.8645

The submission was an honest fine-tune of the provided base: all 51 weight
matrices at cosine >= 0.9999 (median 1.0000), and exactly 1 of 100 tensors under
the floor -- a 768-element attention key bias whose near-zero entries rotate a
long way under a functionally irrelevant update.

Fixed in commit 8d42c88: the cosine floor now applies to weight matrices only,
and 1-D cosines are reported as `min_vector_cosine` rather than used to reject.
