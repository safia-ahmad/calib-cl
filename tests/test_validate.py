import numpy as np

from calcl.validate import known_answer_table


def test_known_answers_on_a_calibrated_model():
    # 40 slides, a well-calibrated model with good but imperfect separation
    rng = np.random.default_rng(0)
    n = 4000
    d = rng.normal(0, 2.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-d))).astype(float)
    groups = np.repeat(np.arange(40), n // 40)
    tab = known_answer_table(d, y, groups, n_rep=5)
    assert (tab[tab.check != "-"]["pass"] == "PASS").all(), tab.to_string()


def test_underconfident_start_is_centred_first():
    # model whose logits are too small (underconfident): sharpening would *help* it,
    # so the check must first make the start calibrated
    rng = np.random.default_rng(1)
    n = 1840
    d = rng.normal(0, 2.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.4 * d + 0.3))).astype(float)
    groups = np.repeat(np.arange(40), n // 40)
    tab = known_answer_table(d, y, groups, n_rep=5)
    assert (tab[tab.check != "-"]["pass"] == "PASS").all(), tab.to_string()
