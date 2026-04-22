"""
Ensemble Fusion Strategies for Combining Positive and Negative Model Predictions
=================================================================================

Multiple strategies for combining predictions from:
  - Positive model: "Is this SUSPICIOUS?" (output: P(suspicious))
  - Negative model: "Is this NON-SUSPICIOUS?" (output: P(non-suspicious))

Goal: Reduce false positives through disagreement detection and agreement voting.
"""

import numpy as np
from enum import Enum


class FusionStrategy(Enum):
    """Available ensemble fusion strategies"""
    AVERAGE = "average"                      # Simple average of both probabilities
    AGREEMENT_VOTING = "agreement_voting"    # Both must agree strongly above threshold
    WEIGHTED_AVERAGE = "weighted_average"    # Weight by individual ROC-AUC scores
    CONTRADICTION = "contradiction_score"    # Higher score when models agree, lower when conflict
    MAX_MIN_VOTING = "max_min_voting"        # Use max for suspicious, min for safety margin
    BAYESIAN_FUSION = "bayesian"             # Bayesian combination using calibration stats


def simple_average(p_suspicious: float, p_not_suspicious: float) -> dict:
    """
    Strategy 1: Simple Average
    
    Both predictions averaged directly.
    
    Args:
        p_suspicious: Probability from positive model (0-1, higher = suspicious)
        p_not_suspicious: Probability from negative model (0-1, higher = non-suspicious)
    
    Returns:
        dict with decision, confidence, reasoning
    """
    # Convert negative model to suspicious scale: 1 - P(non-susp) = P(susp)
    p_susp_from_negative = 1.0 - p_not_suspicious
    
    # Average both estimates of P(suspicious)
    combined = (p_suspicious + p_susp_from_negative) / 2.0
    
    return {
        'combined_confidence': combined,
        'p_positive_model': p_suspicious,
        'p_negative_model': p_not_suspicious,
        'strategy': 'simple_average',
        'reasoning': f'Avg of {p_suspicious:.3f} (positive) and {p_susp_from_negative:.3f} (negative→positive) = {combined:.3f}'
    }


def agreement_voting(p_suspicious: float, p_not_suspicious: float, 
                     thresh_positive=0.5, thresh_negative=0.5) -> dict:
    """
    Strategy 2: Agreement Voting
    
    Only flag as suspicious if BOTH models agree:
    - Positive model says YES (> thresh_positive)
    - Negative model says NO (< 1 - thresh_negative, i.e., P(non-susp) low)
    
    Strongly reduces false positives when models disagree.
    
    Args:
        p_suspicious: Probability from positive model
        p_not_suspicious: Probability from negative model
        thresh_positive: Threshold for positive model to flag as suspicious
        thresh_negative: Threshold for negative model's confidence in non-suspicious
    
    Returns:
        dict with decision, confidence, reasoning
    """
    p_susp_from_neg = 1.0 - p_not_suspicious
    
    # Strong agreement?
    positive_votes_suspicious = p_suspicious > thresh_positive
    negative_votes_suspicious = p_susp_from_neg > (1 - thresh_negative)
    
    agreement_level = 0.0
    if positive_votes_suspicious and negative_votes_suspicious:
        agreement_level = min(p_suspicious, p_susp_from_neg)
        reasoning = f"✅ STRONG AGREEMENT: both models say suspicious ({p_suspicious:.3f}, {p_susp_from_neg:.3f})"
    elif not positive_votes_suspicious and not negative_votes_suspicious:
        agreement_level = min(1 - p_suspicious, 1 - p_susp_from_neg)
        reasoning = f"✅ STRONG AGREEMENT: both models say non-suspicious ({p_suspicious:.3f}, {p_susp_from_neg:.3f})"
    else:
        agreement_level = 0.5  # Disagreement: uncertain
        reasoning = f"⚠️ DISAGREEMENT: positive={p_suspicious:.3f}, negative={p_susp_from_neg:.3f} (conflicting signals)"
    
    return {
        'combined_confidence': agreement_level,
        'p_positive_model': p_suspicious,
        'p_negative_model': p_not_suspicious,
        'strategy': 'agreement_voting',
        'reasoning': reasoning,
        'agreement_score': agreement_level
    }


def weighted_average(p_suspicious: float, p_not_suspicious: float,
                    positive_weight=0.5, negative_weight=0.5) -> dict:
    """
    Strategy 3: Weighted Average
    
    Weight each model's prediction based on validation ROC-AUC (or other calibration metric).
    Better-performing models get higher weight.
    
    Typical: model_A AUC=0.607, model_B AUC=0.61 → weights roughly equal
    
    Args:
        p_suspicious: Probability from positive model
        p_not_suspicious: Probability from negative model
        positive_weight: Weight for positive model (0-1)
        negative_weight: Weight for negative model (0-1)
    
    Returns:
        dict with decision, confidence, reasoning
    """
    if positive_weight + negative_weight != 1.0:
        # Normalize
        total = positive_weight + negative_weight
        positive_weight /= total
        negative_weight /= total
    
    p_susp_from_neg = 1.0 - p_not_suspicious
    combined = (p_suspicious * positive_weight) + (p_susp_from_neg * negative_weight)
    
    return {
        'combined_confidence': combined,
        'p_positive_model': p_suspicious,
        'p_negative_model': p_not_suspicious,
        'strategy': 'weighted_average',
        'weights': {'positive': positive_weight, 'negative': negative_weight},
        'reasoning': f'Weighted: {p_suspicious:.3f}×{positive_weight:.2f} + {p_susp_from_neg:.3f}×{negative_weight:.2f} = {combined:.3f}'
    }


def contradiction_score(p_suspicious: float, p_not_suspicious: float) -> dict:
    """
    Strategy 4: Contradiction Score (most clinically interpretable)
    
    Penalizes when models disagree, helps catch hallucinations.
    
    - If both agree (both say suspicious OR both say non-suspicious):
      confidence = higher of the two
    - If they disagree (one says suspicious, other non-suspicious):
      confidence = reduced, indicates uncertain case needing human review
    
    Formula:
      p_susp_from_neg = 1 - P(non-suspicious)
      agreement = 1 - |P(suspicious) - P(susp from neg)|
      if both in high confidence range: use their average
      if conflicted: confidence = lower of the two (be conservative)
    
    Args:
        p_suspicious: Probability from positive model
        p_not_suspicious: Probability from negative model
    
    Returns:
        dict with decision, confidence, reasoning, and contradiction measure
    """
    p_susp_from_neg = 1.0 - p_not_suspicious
    
    # Disagreement magnitude
    disagreement = abs(p_suspicious - p_susp_from_neg)
    agreement_metric = 1.0 - disagreement  # 1.0 = perfect agreement, 0.0 = total conflict
    
    # If high disagreement (>0.3), be conservative: use minimum probability
    if disagreement > 0.3:
        combined = min(p_suspicious, p_susp_from_neg)
        confidence_type = "CONSERVATIVE (models conflict)"
    else:
        # Models roughly agree, use average
        combined = (p_suspicious + p_susp_from_neg) / 2.0
        confidence_type = "CONFIDENT (models agree)"
    
    return {
        'combined_confidence': combined,
        'p_positive_model': p_suspicious,
        'p_negative_model': p_not_suspicious,
        'strategy': 'contradiction_score',
        'disagreement_magnitude': disagreement,
        'agreement_metric': agreement_metric,
        'confidence_type': confidence_type,
        'reasoning': f"{confidence_type}: P+=({p_suspicious:.3f}, P-={p_susp_from_neg:.3f}), disagreement={disagreement:.3f}, final={combined:.3f}"
    }


def max_min_voting(p_suspicious: float, p_not_suspicious: float) -> dict:
    """
    Strategy 5: Max-Min Voting (medical screening focused)
    
    Combine "aggressive" and "conservative" estimates:
    - Use MAX from positive model (don't miss suspicious)
    - Use MIN from negative model check (don't over-flag safe)
    
    Result: biased toward flagging suspicious cases while filtering obvious false alarms.
    
    Args:
        p_suspicious: Probability from positive model
        p_not_suspicious: Probability from negative model
    
    Returns:
        dict with decision, confidence, reasoning
    """
    p_susp_from_neg = 1.0 - p_not_suspicious
    
    # Use max from positive (aggressive for finding suspicious)
    # and min from negative reframed (conservative filter for safety)
    # Idea: if negative is very confident it's non-suspicious (high p_not_suspicious),
    # we trust that safety signal
    
    safety_margin = p_not_suspicious  # High = model confident it's safe
    suspicious_signal = p_suspicious  # High = model confident it's suspicious
    
    # Combined: be aggressive on suspicious, but reduce confidence if safety signal is strong
    combined = suspicious_signal * (1.0 - safety_margin * 0.5)
    
    return {
        'combined_confidence': combined,
        'p_positive_model': p_suspicious,
        'p_negative_model': p_not_suspicious,
        'strategy': 'max_min_voting',
        'aggressive_signal': suspicious_signal,
        'safety_margin': safety_margin,
        'reasoning': f'Aggressive suspicious signal {suspicious_signal:.3f} × (1.0 - safety margin {safety_margin:.3f}/2) = {combined:.3f}'
    }


def bayesian_fusion(p_suspicious: float, p_not_suspicious: float,
                   prior_susp=0.3,  # Prior: ~30% of images are suspicious in clinic
                   pos_model_calibration=0.607,  # Positive model ROC-AUC
                   neg_model_calibration=0.61) -> dict:
    """
    Strategy 6: Bayesian Fusion
    
    Treat model outputs as likelihoods in Bayesian framework.
    
    P(suspicious | both models) ∝ P(models | suspicious) × P(suspicious)
    
    Requires model calibration (ROC-AUC or similar) to convert to likelihood ratios.
    
    Args:
        p_suspicious: Output from positive model
        p_not_suspicious: Output from negative model
        prior_susp: Prior probability of suspicious (from prevalence in training set)
        pos_model_calibration: Positive model calibration metric (e.g., ROC-AUC)
        neg_model_calibration: Negative model calibration metric
    
    Returns:
        dict with Bayesian posterior and reasoning
    """
    p_susp_from_neg = 1.0 - p_not_suspicious
    
    # Simple Bayesian update: both models vote
    # Likelihood ratio: how much does each model's output update our belief?
    lr_positive = p_suspicious / (1.0 - p_suspicious + 1e-6) if p_suspicious != 0.5 else 1.0
    lr_negative = p_susp_from_neg / (1.0 - p_susp_from_neg + 1e-6) if p_susp_from_neg != 0.5 else 1.0
    
    # Combined likelihood ratio
    combined_lr = lr_positive * lr_negative
    
    # Convert back to probability using prior
    posterior = (combined_lr * (prior_susp / (1.0 - prior_susp))) / (
        1.0 + combined_lr * (prior_susp / (1.0 - prior_susp))
    )
    posterior = np.clip(posterior, 0.0, 1.0)
    
    return {
        'combined_confidence': posterior,
        'p_positive_model': p_suspicious,
        'p_negative_model': p_not_suspicious,
        'strategy': 'bayesian',
        'prior': prior_susp,
        'likelihood_ratio': combined_lr,
        'posterior': posterior,
        'reasoning': f'Bayesian posterior (prior={prior_susp:.2f}, LR={combined_lr:.3f}) = {posterior:.3f}'
    }


def ensemble_predict(p_suspicious: float, p_not_suspicious: float,
                    strategies=['agreement_voting', 'contradiction_score', 'weighted_average'],
                    threshold=0.5) -> dict:
    """
    Run multiple ensemble strategies and return all results.
    
    Args:
        p_suspicious: Output from positive model (0-1)
        p_not_suspicious: Output from negative model (0-1)
        strategies: List of strategy names to run
        threshold: Decision threshold for final classification
    
    Returns:
        dict with all strategy results, average decision, and recommendation
    """
    strategy_results = {}
    
    for strat_name in strategies:
        if strat_name == 'average':
            strategy_results[strat_name] = simple_average(p_suspicious, p_not_suspicious)
        elif strat_name == 'agreement_voting':
            strategy_results[strat_name] = agreement_voting(p_suspicious, p_not_suspicious)
        elif strat_name == 'weighted_average':
            strategy_results[strat_name] = weighted_average(p_suspicious, p_not_suspicious, 
                                                           positive_weight=0.5, negative_weight=0.5)
        elif strat_name == 'contradiction_score':
            strategy_results[strat_name] = contradiction_score(p_suspicious, p_not_suspicious)
        elif strat_name == 'max_min_voting':
            strategy_results[strat_name] = max_min_voting(p_suspicious, p_not_suspicious)
        elif strat_name == 'bayesian':
            strategy_results[strat_name] = bayesian_fusion(p_suspicious, p_not_suspicious)
    
    # Average confidence across all strategies
    confidences = [r['combined_confidence'] for r in strategy_results.values()]
    avg_confidence = np.mean(confidences)
    
    # Voting: how many strategies flag as suspicious?
    votes_suspicious = sum(1 for conf in confidences if conf > threshold)
    total_strategies = len(confidences)
    
    final_decision = "SUSPICIOUS" if avg_confidence > threshold else "NON-SUSPICIOUS"
    
    return {
        'individual_strategies': strategy_results,
        'average_confidence': avg_confidence,
        'votes_suspicious': votes_suspicious,
        'total_strategies': total_strategies,
        'consensus_strength': votes_suspicious / total_strategies,
        'final_decision': final_decision,
        'summary': f"{votes_suspicious}/{total_strategies} strategies agree: {final_decision} (avg confidence={avg_confidence:.3f})"
    }
