"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# See PDF, Part 1 Q1.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.array([[3, 4], [2, 1]], dtype=float) if A is None else np.asarray(A, dtype=float)
    B = np.array([[1, 2], [0, -1]], dtype=float) if B is None else np.asarray(B, dtype=float)
    AB = A @ B
    BA = B @ A
    return AB, BA, (A + B).T, A @ B.T, float(np.trace(A)), float(np.trace(AB))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# See PDF, Part 1 Q2.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.array([[2, -1], [-1, 2]], dtype=float) if M is None else np.asarray(M, dtype=float)
    values, vectors = np.linalg.eigh(M)
    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# See PDF, Part 1 Q3.

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# See PDF, Part 1 Q4.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.array([[4, 7], [2, 6]], dtype=float) if C is None else np.asarray(C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# See PDF, Part 2 Q1.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    rate = float(rate)
    return float(np.exp(-rate)), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# See PDF, Part 2 Q2.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    inter = p_a_given_b * p_b
    return inter, inter / p_a, p_a + p_b - inter


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# See PDF, Part 2 Q3.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# See PDF, Part 2 Q4.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    x = np.array([1, 3, 5, 7], dtype=float) if data is None else np.asarray(data, dtype=float)
    mu = x.mean()
    return float(mu), float(np.mean((x - mu) ** 2))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# See PDF, Part 2 Q5.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    precision = 1.0 / observation_variance + 1.0 / prior_variance
    return (x / observation_variance + prior_mean / prior_variance) / precision


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# See PDF, Part 2 Q6.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# See PDF, Part 3 Q1.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# See PDF, Part 3 Q2.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.array([2.0, -1.0]) if mean is None else np.array(mean, dtype=float)
    covariance = (np.array([[3.0, 1.2], [1.2, 1.0]]) if covariance is None
                  else np.array(covariance, dtype=float))
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# See PDF, Part 3 Q3.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    mean = X.mean(axis=0)
    centered = X - mean                      # new array; X is untouched
    covariance = centered.T @ centered / (n - 1)
    covariance = (covariance + covariance.T) / 2   # remove round-off asymmetry
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]    # descending
    eigenvalues = np.maximum(eigenvalues[order], 0.0)   # clip round-off negatives
    components = eigenvectors[:, order]
    scores = centered @ components
    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# See PDF, Part 3 Q4.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# See PDF, Part 3 Q2-Q4.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    X = np.asarray(X, dtype=float)
    mean = result["mean"]
    lam = result["eigenvalues"]
    V = result["components"]
    frac = explained_variance(lam)
    rho = np.corrcoef(X[:, 0], X[:, 1])[0, 1] if X.shape[0] > 1 else np.nan

    # Figure 1: raw data
    fig1 = Figure(figsize=(5, 4.2), constrained_layout=True)
    ax = fig1.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5, color="tab:blue")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"Synthetic 2-D Gaussian data (r = {rho:.2f})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

    # Figure 2: data with principal directions
    fig2 = Figure(figsize=(5, 4.2), constrained_layout=True)
    ax = fig2.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.35, color="gray", label="samples")
    colors = ["tab:red", "tab:green"]
    for i in range(2):
        v = V[:, i] if V[np.argmax(np.abs(V[:, i])), i] > 0 else -V[:, i]  # display sign only
        d = v * 2.0 * np.sqrt(lam[i])
        ax.annotate("", xy=mean + d, xytext=mean,
                    arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=2.5))
        ax.plot([], [], color=colors[i], lw=2.5,
                label=f"PC{i + 1} ({100 * frac[i]:.1f}% var)")
    ax.scatter(*mean, color="black", marker="x", s=60, zorder=5, label="mean")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Principal directions through the mean")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Figure 3: explained variance
    fig3 = Figure(figsize=(5, 4.2), constrained_layout=True)
    ax = fig3.subplots()
    bars = ax.bar(["PC1", "PC2"], frac, color=colors)
    ax.plot(["PC1", "PC2"], np.cumsum(frac), "ko--", label="cumulative")
    for b, f in zip(bars, frac):
        ax.text(b.get_x() + b.get_width() / 2, f / 2, f"{100 * f:.1f}%",
                ha="center", va="center", color="white", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Proportion of total variance")
    ax.set_title("Explained variance")
    ax.legend(loc="center right")

    return fig1, fig2, fig3


# REFLECTION: contributions, tasks completed, and external resources used.
# This assignment was completed with AI assistance. Claude drafted
# the derivations, code, plots, and write-up. All Part 1-3 tasks are completed.
# External resources: NumPy and Matplotlib documentation only.

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    # Part 1
    AB, BA, ABT_sum, A_BT, trA, trAB = matrix_operations()
    print("AB =\n", AB, "\nBA =\n", BA, "\n(A+B)^T =\n", ABT_sum,
          "\nA B^T =\n", A_BT, "\ntrace(A) =", trA, " trace(AB) =", trAB)
    vals, vecs = eigensystem()
    print("eigenvalues:", vals, "\neigenvectors (columns):\n", vecs)
    M = np.array([[2, -1], [-1, 2]], dtype=float)
    print("Mv == lam v:", np.allclose(M @ vecs, vecs * vals))
    print("C^-1 =\n", matrix_inverse())

    # Part 2
    print("c, mean, var:", distribution_moments())
    print("P(A&B), P(B|A), P(AuB):", event_probabilities())
    print("MLE, MAP coin:", coin_estimates())
    print("Gaussian MLE (mu, sigma^2):", gaussian_mle())
    print("Gaussian MAP:", gaussian_map())

    # Part 3
    X = generate_data()
    res = pca(X)
    frac = explained_variance(res["eigenvalues"])
    print("sample mean:", res["mean"])
    print("covariance:\n", res["covariance"])
    print("eigenvalues:", res["eigenvalues"])
    print("components (columns):\n", res["components"])
    print("explained variance fractions:", frac, " cumulative:", np.cumsum(frac))
    print("correlation:", np.corrcoef(X.T)[0, 1])
    for name, fig in zip(("fig1_data", "fig2_pca", "fig3_variance"), make_plots(X, res)):
        fig.savefig(f"{name}.png", dpi=200)
