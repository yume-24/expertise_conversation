"""Geometry formulas preserved from the paper notebook."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

def _sample_sd(values):
    """Sample standard deviation, or NaN if only one value exists."""
    values = np.asarray(values)

    if values.size < 2:
        return np.nan

    return float(np.std(values, ddof=1))


def calculate_conversation_geometry(embeddings, n_words=None, n_turns=None, k=5):
    """
    Calculate the complete geometric metric set for one conversation.

    Parameters
    ----------
    embeddings : array-like, shape (n_utterances, embedding_dimension)
        One embedding per utterance.

    n_words : int, optional
        Total number of words in the conversation.

    n_turns : int, optional
        Total number of conversational turns.

    k : int, default=5
        Neighbor rank used for the k-nearest-neighbor statistic.

    Returns
    -------
    scalar_metrics : dict
        Conversation-level scalar metrics suitable for a DataFrame.

    spectrum : dict
        Eigenvalue arrays retained separately for spectral plots.
    """

    X = np.asarray(embeddings, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(
            "embeddings must have shape "
            "(n_utterances, embedding_dimension)"
        )

    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    n, d = X.shape

    if n < 2:
        raise ValueError(
            "At least two utterances are required."
        )

    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")

    # ------------------------------------------------------------
    # 1. Normalize embeddings
    # ------------------------------------------------------------

    norms = np.linalg.norm(X, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("At least one embedding has zero norm.")

    X_unit = X / norms

    # ------------------------------------------------------------
    # 2. Pairwise cosine similarity and distance
    # ------------------------------------------------------------

    cosine_similarity_matrix = np.clip(
        X_unit @ X_unit.T,
        -1.0,
        1.0,
    )

    cosine_distance_matrix = 1.0 - cosine_similarity_matrix

    # Set the diagonal to exactly zero for neighbor calculations.
    np.fill_diagonal(cosine_distance_matrix, 0.0)

    upper_triangle = np.triu_indices(n, k=1)

    pairwise_similarities = cosine_similarity_matrix[
        upper_triangle
    ]

    pairwise_distances = cosine_distance_matrix[
        upper_triangle
    ]

    n_pairs = len(pairwise_distances)

    sim_q10, sim_median, sim_q90 = np.quantile(
        pairwise_similarities,
        [0.10, 0.50, 0.90],
    )

    dist_q10, dist_median, dist_q90 = np.quantile(
        pairwise_distances,
        [0.10, 0.50, 0.90],
    )

    # ------------------------------------------------------------
    # 3. Centroid concentration
    # ------------------------------------------------------------

    centroid = X_unit.mean(axis=0)

    centroid_resultant_length = float(
        np.linalg.norm(centroid)
    )

    centroid_dispersion_cosine = float(
        1.0 - centroid_resultant_length
    )

    # ------------------------------------------------------------
    # 4. Radial structure
    # ------------------------------------------------------------

    centroid_tolerance = (
        100 * np.finfo(np.float64).eps
    )

    if centroid_resultant_length > centroid_tolerance:
        centroid_direction = (
            centroid / centroid_resultant_length
        )

        radial_distances = np.clip(
            1.0 - X_unit @ centroid_direction,
            0.0,
            2.0,
        )

        radial_distance_mean = float(
            np.mean(radial_distances)
        )

        radial_distance_median = float(
            np.median(radial_distances)
        )

        radial_distance_q90 = float(
            np.quantile(radial_distances, 0.90)
        )

    else:
        # A zero centroid has no defined direction.
        radial_distances = np.full(n, np.nan)
        radial_distance_mean = np.nan
        radial_distance_median = np.nan
        radial_distance_q90 = np.nan

    # ------------------------------------------------------------
    # 5. Local-neighborhood structure
    # ------------------------------------------------------------

    neighbor_distance_matrix = (
        cosine_distance_matrix.copy()
    )

    # Prevent each utterance from selecting itself.
    np.fill_diagonal(
        neighbor_distance_matrix,
        np.inf,
    )

    nearest_neighbor_distances = np.min(
        neighbor_distance_matrix,
        axis=1,
    )

    k_used = min(k, n - 1)

    kth_neighbor_distances = np.partition(
        neighbor_distance_matrix,
        kth=k_used - 1,
        axis=1,
    )[:, k_used - 1]

    # ------------------------------------------------------------
    # 6. Centered covariance spectrum
    # ------------------------------------------------------------

    X_centered = X_unit - centroid

    singular_values = np.linalg.svd(
        X_centered,
        compute_uv=False,
        full_matrices=False,
    )

    # Covariance eigenvalues:
    # lambda_j = s_j^2 / (n - 1)
    eigenvalues = (
        singular_values**2 / (n - 1)
    )

    total_variance = float(
        eigenvalues.sum()
    )

    maximum_available_rank = min(
        n - 1,
        d,
    )

    leading_singular_value = (
        float(singular_values[0])
        if singular_values.size
        else 0.0
    )

    sv_tolerance = (
        max(n, d)
        * np.finfo(np.float64).eps
        * leading_singular_value
    )

    active = singular_values > sv_tolerance

    # ------------------------------------------------------------
    # 7. Spectral metrics
    # ------------------------------------------------------------

    if np.isclose(
        total_variance,
        0.0,
        atol=1e-15,
        rtol=0.0,
    ):
        numerical_rank = 0
        normalized_eigenvalues = np.array([])

        spectral_entropy = np.nan
        spectral_entropy_normalized = np.nan
        spectral_effective_rank = np.nan
        participation_ratio = np.nan
        pc1_variance_share = np.nan

    else:
        numerical_rank = int(active.sum())

        active_eigenvalues = eigenvalues[active]

        normalized_eigenvalues = (
            active_eigenvalues
            / active_eigenvalues.sum()
        )

        # H = -sum(p_j log p_j)
        spectral_entropy = float(
            -np.sum(
                normalized_eigenvalues
                * np.log(normalized_eigenvalues)
            )
        )

        # Effective rank = exp(H)
        spectral_effective_rank = float(
            np.exp(spectral_entropy)
        )

        # PR = 1 / sum(p_j^2)
        participation_ratio = float(
            1.0
            / np.sum(normalized_eigenvalues**2)
        )

        # Proportion of variance explained by PC1
        pc1_variance_share = float(
            eigenvalues[0] / total_variance
        )

        if maximum_available_rank > 1:
            spectral_entropy_normalized = float(
                spectral_entropy
                / np.log(maximum_available_rank)
            )
        else:
            spectral_entropy_normalized = np.nan

    # ------------------------------------------------------------
    # 8. Collect scalar results
    # ------------------------------------------------------------

    scalar_metrics = {
        # Bookkeeping
        "n_utterances": int(n),
        "n_pairs": int(n_pairs),
        "n_words": (
            int(n_words)
            if n_words is not None
            else np.nan
        ),
        "n_turns": (
            int(n_turns)
            if n_turns is not None
            else np.nan
        ),
        "embedding_dimension": int(d),
        "k_used": int(k_used),

        # Pairwise cosine similarity
        "pairwise_cosine_similarity_mean": float(
            np.mean(pairwise_similarities)
        ),
        "pairwise_cosine_similarity_sd": _sample_sd(
            pairwise_similarities
        ),
        "pairwise_cosine_similarity_q10": float(
            sim_q10
        ),
        "pairwise_cosine_similarity_median": float(
            sim_median
        ),
        "pairwise_cosine_similarity_q90": float(
            sim_q90
        ),

        # Centroid concentration
        "centroid_resultant_length": (
            centroid_resultant_length
        ),
        "centroid_dispersion_cosine": (
            centroid_dispersion_cosine
        ),

        # Radial structure
        # The mean is retained mainly as an identity check.
        "radial_distance_mean_check": (
            radial_distance_mean
        ),
        "radial_distance_median": (
            radial_distance_median
        ),
        "radial_distance_q90": (
            radial_distance_q90
        ),

        # Pairwise cosine distance
        "pairwise_distance_mean": float(
            np.mean(pairwise_distances)
        ),
        "pairwise_distance_sd": _sample_sd(
            pairwise_distances
        ),
        "pairwise_distance_q10": float(
            dist_q10
        ),
        "pairwise_distance_median": float(
            dist_median
        ),
        "pairwise_distance_q90": float(
            dist_q90
        ),
        "pairwise_semantic_span": float(
            dist_q90 - dist_q10
        ),

        # Local-neighborhood structure
        "nearest_neighbor_distance_mean": float(
            np.mean(nearest_neighbor_distances)
        ),
        "nearest_neighbor_distance_median": float(
            np.median(nearest_neighbor_distances)
        ),
        "knn_distance_median": float(
            np.median(kth_neighbor_distances)
        ),

        # Covariance-spectrum structure
        "maximum_available_rank": int(
            maximum_available_rank
        ),
        "numerical_rank": int(
            numerical_rank
        ),
        "total_variance": (
            total_variance
        ),
        "spectral_entropy": (
            spectral_entropy
        ),
        "spectral_entropy_normalized": (
            spectral_entropy_normalized
        ),
        "spectral_effective_rank": (
            spectral_effective_rank
        ),
        "participation_ratio": (
            participation_ratio
        ),
        "pc1_variance_share": (
            pc1_variance_share
        ),
    }

    # Retain arrays separately so they do not clutter the DataFrame.
    spectrum = {
        "covariance_eigenvalues": eigenvalues,
        "normalized_eigenvalues": normalized_eigenvalues,
    }

    # ------------------------------------------------------------
    # 9. Mathematical consistency checks
    # ------------------------------------------------------------

    np.testing.assert_allclose(
        scalar_metrics["pairwise_distance_mean"],
        total_variance,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        scalar_metrics[
            "pairwise_cosine_similarity_mean"
        ],
        1.0
        - scalar_metrics["pairwise_distance_mean"],
        atol=1e-10,
    )

    np.testing.assert_allclose(
        scalar_metrics["pairwise_distance_mean"],
        n / (n - 1)
        * (1.0 - centroid_resultant_length**2),
        atol=1e-10,
    )

    if np.isfinite(radial_distance_mean):
        np.testing.assert_allclose(
            radial_distance_mean,
            centroid_dispersion_cosine,
            atol=1e-10,
        )

    return scalar_metrics, spectrum


def calculate_speaker_centroid_distance(
    conversation_utterances,
    embeddings,
    role_column="_centroid_role",
    embedding_index_column="embedding_idx",
):
    """Return expert–partner centroid distances for one conversation."""
    expert_indices = (
        conversation_utterances.loc[
            conversation_utterances[role_column].eq("expert"),
            embedding_index_column,
        ]
        .dropna()
        .astype(int)
        .to_numpy()
    )

    partner_indices = (
        conversation_utterances.loc[
            conversation_utterances[role_column].eq("partner"),
            embedding_index_column,
        ]
        .dropna()
        .astype(int)
        .to_numpy()
    )

    if len(expert_indices) == 0 or len(partner_indices) == 0:
        raise ValueError(
            "Each conversation must contain both expert and partner utterances."
        )

    expert_centroid = embeddings[expert_indices].mean(axis=0)
    partner_centroid = embeddings[partner_indices].mean(axis=0)

    return {
        "expert_n_utterances": int(len(expert_indices)),
        "partner_n_utterances": int(len(partner_indices)),
        "speaker_centroid_cosine_distance": float(
            cosine(expert_centroid, partner_centroid)
        ),
        "speaker_centroid_euclidean_distance": float(
            np.linalg.norm(expert_centroid - partner_centroid)
        ),
    }


def calculate_all_metrics(utterances, embeddings, centroid_min_words=5):
    geometry_utterances = utterances.copy()
    E_utterances = embeddings
    geometry_records = []
    conversation_spectra = {}

    group_columns = [
        "dataset",
        "conversation_id",
    ]

    for group_key, group in geometry_utterances.groupby(
        group_columns,
        sort=False,
        observed=True,
    ):
        dataset, conversation_id = group_key

        # embedding_idx maps each utterance row to E_utterances.
        embedding_indices = group[
            "embedding_idx"
        ].to_numpy(dtype=int)

        conversation_embeddings = E_utterances[
            embedding_indices
        ]

        scalar_metrics, spectrum = (
            calculate_conversation_geometry(
                conversation_embeddings,
                n_words=group["n_words"].sum(),
                n_turns=group["turn_id"].nunique(),
                k=5,
            )
        )

        row = {
            "dataset": dataset,
            "conversation_id": conversation_id,
        }

        # Preserve useful conversation metadata where available.
        for column in [
            "video_id",
            "file_number",
            "level",
            "level_label",
        ]:
            if column in group.columns:
                row[column] = group[column].iloc[0]

        row.update(scalar_metrics)
        geometry_records.append(row)

        conversation_spectra[
            (dataset, conversation_id)
        ] = spectrum


    conversation_geometry = pd.DataFrame(
        geometry_records
    )

    sort_columns = [
        column
        for column in [
            "level",
            "dataset",
            "conversation_id",
        ]
        if column in conversation_geometry.columns
    ]

    conversation_geometry = (
        conversation_geometry
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )


    centroid_rows = utterances.loc[utterances.n_words.ge(centroid_min_words)].copy()
    centroid_rows['_centroid_role'] = centroid_rows.speaker_role
    records = []
    for (dataset, conversation_id), group in centroid_rows.groupby(['dataset', 'conversation_id'], observed=True, sort=False):
        result = calculate_speaker_centroid_distance(group, embeddings)
        records.append({'dataset': dataset, 'conversation_id': conversation_id, **result})
    centroid_frame = pd.DataFrame(records)
    if centroid_frame.empty:
        raise ValueError('No eligible speaker-centroid utterances')
    combined = conversation_geometry.merge(centroid_frame, on=['dataset', 'conversation_id'], how='left', validate='one_to_one')
    from .plots import METRIC_LABELS
    columns = ['dataset', 'conversation_id', 'video_id', 'file_number', 'level', 'level_label',
               'n_utterances', 'expert_n_utterances', 'partner_n_utterances', *METRIC_LABELS]
    metrics = combined[columns].copy()
    if metrics.duplicated(['dataset', 'conversation_id']).any() or not np.isfinite(metrics[list(METRIC_LABELS)]).all().all():
        raise ValueError('Duplicate conversations or undefined metrics; check speaker eligibility')
    coverage = metrics.groupby(['dataset', 'video_id'], observed=True)['level'].agg(set)
    if not coverage.map(lambda levels: levels == set(range(5))).all():
        raise ValueError('Incomplete five-level video sets')
    if centroid_min_words == 0:
        if not (metrics.expert_n_utterances + metrics.partner_n_utterances).eq(metrics.n_utterances).all():
            raise ValueError('Speaker counts must cover all nonempty utterances')
    return metrics
