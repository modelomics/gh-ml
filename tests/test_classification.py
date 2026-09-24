from gh_ml.classification import classify_repository


def test_ambiguous_abbreviations_and_generic_control_do_not_create_domain_tags():
    labels = classify_repository(
        {
            "name": "rl-control-study",
            "description": "Analysis of the RL control group in a clinical trial",
        },
        [],
    )

    assert "reinforcement-learning" not in labels["domains"]
    assert "robotics-and-control" not in labels["domains"]


def test_specific_robotics_and_reinforcement_learning_terms_are_recognized():
    labels = classify_repository(
        {"description": "Reinforcement learning for robot control policies"}, [],
    )

    assert "reinforcement-learning" in labels["domains"]
    assert "robotics-and-control" in labels["domains"]


def test_applied_domains_and_classic_methods_are_open_vocabulary_labels():
    labels = classify_repository(
        {
            "description": "Crop disease image classification with convolutional neural networks",
            "topics": ["precision-agriculture"],
        },
        [],
    )

    assert "agriculture-and-food" in labels["domains"]
    assert "computer-vision" in labels["domains"]
    assert "convolutional neural network" in labels["methods"]


def test_query_labels_remain_open_vocabulary():
    labels = classify_repository(
        {}, [{"domains": ["computational-linguistics"], "methods": ["new-technique"]}],
    )

    assert labels["domains"] == ["computational-linguistics"]
    assert labels["methods"] == ["new-technique"]
