# Reusable A/B testing helpers from the Cookie Cats Bayesian A/B testing notebook
import numpy as np
from scipy import stats

N_DRAWS = 200_000


def ztest_two_prop(x_a, n_a, x_b, n_b, alpha=0.05):
  p_a, p_b = x_a / n_a, x_b / n_b
  p_pool = (x_a + x_b) / (n_a + n_b)
  se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
  z = (p_b - p_a) / se_pool # a negative difference means the new gate placement is worse
  p_value = 2 * stats.norm.sf(abs(z)) # survival func
  se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)   # standard error, unpooled for the CI
  zc = stats.norm.ppf(1 - alpha / 2) # 95% CI with alpha 0.05
  diff = p_b - p_a
  return dict(p_A=p_a, p_B=p_b, diff=diff, ci_low=diff - zc * se,
              ci_high=diff + zc * se, z=z, p_value=p_value)


def bayesian_ab(x_a, n_a, x_b, n_b, prior=(1, 1), draws=N_DRAWS, seed=0):
  # Beta-Binomial A/B test. Returns summary dict and posterior samples.
  r = np.random.default_rng(seed)
  a0, b0 = prior
  pa = r.beta(a0 + x_a, b0 + n_a - x_a, draws)
  pb = r.beta(a0 + x_b, b0 + n_b - x_b, draws)
  diff = pb - pa
  lift = diff / pa # relative lift of B over A
  out = dict(
    mean_A=pa.mean(), mean_B=pb.mean(),
    prob_B_better=(pb > pa).mean(),
    diff_mean=diff.mean(),
    diff_ci_low=np.quantile(diff, 0.025), diff_ci_high=np.quantile(diff, 0.975),
    lift_mean=lift.mean(),
    lift_ci_low=np.quantile(lift, 0.025), lift_ci_high=np.quantile(lift, 0.975),
    loss_choose_B=np.maximum(pa - pb, 0).mean(),
    loss_choose_A=np.maximum(pb - pa, 0).mean(),
  )
  return out, dict(pa=pa, pb=pb, diff=diff, lift=lift)


def bayes_bootstrap(x, n_rep=2000, seed=0):
  # Posterior draws of the mean and median via Dirichlet-weighted resampling.
  # Players with the same value are grouped so each replicate is cheap.
  r = np.random.default_rng(seed)
  vals, count = np.unique(x, return_counts=True)
  means, medians = np.empty(n_rep), np.empty(n_rep) # (count) players played (vals) round
  for i in range(n_rep):
    # Sum of `count` iid Exp(1) is Gamma(count, 1): equivalent to Dirichlet weights per player
    w = r.gamma(count, 1.0)
    w /= w.sum()
    means[i] = (w * vals).sum()
    medians[i] = vals[np.searchsorted(np.cumsum(w), 0.5)]
  return means, medians
