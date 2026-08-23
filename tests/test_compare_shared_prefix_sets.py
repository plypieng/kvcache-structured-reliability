import math

from toy_kv_experiments.compare_shared_prefix_sets import compare


def summary(rate, structural, other):
    return {
        "tasks": 2,
        "positions": 10,
        "top1_disagreement_rate": rate,
        "structural_top1_disagreement_rate": structural,
        "other_top1_disagreement_rate": other,
    }


def test_compare_reports_difference_and_ratio():
    report = compare(summary(0.2, 0.3, 0.1), summary(0.1, 0.1, 0.1))

    assert report["failure_minus_control"]["top1_disagreement_rate"] == 0.1
    assert math.isclose(
        report["failure_over_control"]["structural_top1_disagreement_rate"],
        3.0,
    )
