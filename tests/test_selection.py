from __future__ import annotations

from gh_ml.selection import SELECTION_VERSION, assess_repository


def test_archived_zipline_is_excluded_as_plain_backtesting_utility() -> None:
    # Repository-owned fields copied from the archived 2026-09-24 current view.
    row = {
        "name": "quantopian/zipline",
        "description": "Zipline, a Pythonic Algorithmic Trading Library",
        "topics": ["algorithmic-trading", "python", "quant", "zipline"],
        "fork": False,
    }
    result = assess_repository(row)
    assert result["selection_status"] == "exclude"
    assert result["selection_reason"] == "non-ml-utility"


def test_fork_of_zipline_is_excluded() -> None:
    assert assess_repository({
        "name": "aichi/zipline",
        "description": "Zipline, a Pythonic Algorithmic Trading Library",
        "fork": True,
    })["selection_reason"] == "fork"


def test_owner_profile_repository_is_excluded() -> None:
    for row in (
        {"name": "octocat", "full_name": "octocat/octocat"},
        {"name": "octocat", "full_name": "some-org/.github"},
        {"name": "octocat/octocat"},
    ):
        result = assess_repository({**row, "description": "Machine learning"})
        assert result["selection_status"] == "exclude"
        assert result["selection_reason"] == "owner-profile-repository"


def test_coursework_and_awesome_lists_are_excluded() -> None:
    for row in (
        {"name": "student/CS760-project", "description": "Transformer homework"},
        {"name": "student/assignment", "description": "CS760 coursework: implement a neural network"},
        {"name": "student/ps3", "description": "CS224N assignment: implement a BERT model"},
        {"name": "student/project", "description": "EECS498 class project using diffusion"},
        {"name": "curator/awesome-ml", "topics": ["awesome-list", "machine-learning"]},
    ):
        assert assess_repository(row)["selection_status"] == "exclude"


def test_method_and_substantive_contribution_are_required_for_include() -> None:
    result = assess_repository({
        "name": "lab/new-selective-scan",
        "description": "We propose a new Mamba selective scan architecture for long sequence modeling.",
        "topics": ["deep-learning"],
        "stargazers_count": 0,
    })
    assert result["selection_status"] == "include"
    assert result["selection_version"] == SELECTION_VERSION == "ml-contribution-v3"
    assert result["selection_signals"] == sorted(result["selection_signals"])


def test_generic_ml_or_query_match_alone_never_includes() -> None:
    result = assess_repository({
        "name": "user/interesting-project",
        "description": "A useful machine learning project",
        "query_ids": ["research.transformer"],
        "methods": ["transformer"],
        "all_methods": ["transformer"],
    })
    assert result["selection_status"] == "review"


def test_paper_code_alone_goes_to_review() -> None:
    result = assess_repository({
        "name": "lab/paper-implementation",
        "description": "Code for a diffusion model paper, arXiv:2401.12345.",
    })
    assert result["selection_status"] == "review"
    assert "paper-and-code-cue" in result["selection_signals"]


def test_named_method_implementation_without_novelty_goes_to_review() -> None:
    for description in (
        "Implementation of BERT for text classification.",
        "A transformer training wrapper with batching and logging.",
    ):
        assert assess_repository({"name": "lab/tool", "description": description})["selection_status"] == "review"


def test_method_topic_does_not_pair_with_unrelated_cli_description() -> None:
    result = assess_repository({
        "name": "lab/cli-tool",
        "description": "A command line interface for organizing files.",
        "topics": ["transformer"],
    })
    assert result["selection_status"] == "review"
    assert "method-tied-novelty-claim" not in result["selection_signals"]


def test_tutorial_is_excluded_even_when_description_mentions_novelty() -> None:
    result = assess_repository({
        "name": "lab/new-mamba-method",
        "description": "We propose a novel Mamba architecture. A tutorial explains its use.",
    })
    assert result["selection_status"] == "exclude"


def test_archived_educational_and_book_false_positives_are_excluded() -> None:
    rows = (
        {"name": "pb1672/ML_Projects", "description": "Andrew Ng's Machine Learning Class Projects: Gradient Descent, SVM, Neural Networks"},
        {"name": "raviraju/AI_HomeWork2", "description": "Mancala game implementing MiniMax and AlphaBeta pruning"},
        {"name": "laermannjan/nip-deeprl-project", "description": "Student project in deep reinforcement learning"},
        {"name": "abdur75648/Deep-Learning-Specialization-Coursera", "description": "Assignments and labs for the specialization"},
        {"name": "SergeiVKalinin/AutomatedExperiment_Summer2023", "description": "The summer training course on Bayesian Optimization"},
        {"name": "Gerard-Devlin/NTU-EE5184", "description": "Reference implementation for assignments in the NTU deep learning course"},
        {"name": "arm-education/Advanced-AI-Mixture-of-Experts", "description": "Hands-on course materials to implement Mixture of Experts models"},
        {"name": "arm-education/Advanced-AI-Hardware-Software-Co-Design", "description": "Course materials for quantization and on-device LLM deployment"},
        {"name": "Parvptl/Introduction_to_Machine_Learning", "description": "A collection of lab assignments for an Introduction to Machine Learning course"},
        {"name": "datawhalechina/diy-llm", "description": "Transformer and MoE topics with progressive code assignments for learning"},
        {"name": "anakin87/llm-rl-environments-lil-course", "description": "A little course on reinforcement learning environments"},
        {"name": "LLMBook-zh/LLMBook-zh.github.io", "description": "Large language model book"},
    )
    for row in rows:
        assert assess_repository(row)["selection_status"] == "exclude", row["name"]


def test_novelty_claim_must_be_tied_to_the_method() -> None:
    result = assess_repository({
        "name": "lab/transformer-tool",
        "description": "We propose a novel command line interface for managing experiments.",
        "topics": ["transformer"],
    })
    assert result["selection_status"] == "review"
    assert "method-tied-novelty-claim" not in result["selection_signals"]


def test_github_io_project_pages_are_not_blanket_excluded() -> None:
    result = assess_repository({
        "name": "lab/new-mamba-method.github.io",
        "description": "We propose a novel Mamba architecture for efficient sequence modeling.",
    })
    assert result["selection_status"] == "include"


def test_archive_workflow_index_and_skill_noise_is_excluded() -> None:
    rows = (
        {
            "name": "actypedef/block-scaled-ptq-research",
            "description": "A Codex skill for designing, implementing, and evaluating post-training quantization research with block-scaled formats.",
            "topics": ["codex-skill", "llm-inference", "quantization"],
        },
        {
            "name": "ericluo04/claude-academic-workflow",
            "description": "Academic research workflow for Claude Code: skills covering causal inference, paper reading, literature review, and replication.",
            "topics": ["academic-research", "agentic-workflows", "econometrics"],
        },
        {
            "name": "spydaz/openai4s",
            "description": "This repository indexes open-source AI for Science projects on GitHub, grouped by scientific workflow.",
        },
    )
    for row in rows:
        assert assess_repository(row)["selection_status"] == "exclude", row["name"]


def test_standard_classifier_combo_does_not_count_as_novel() -> None:
    result = assess_repository({
        "name": "Joydrip/Hybrid-ML-Fault-Detection-Electrical-Grids",
        "description": "Developed a hybrid machine learning model for fault detection by combining SVM and Random Forest classifiers.",
    })
    assert result["selection_status"] == "review"
    assert "method-tied-novelty-claim" not in result["selection_signals"]


def test_ambiguous_quantization_needs_ml_context_and_non_ml_filter_is_excluded() -> None:
    result = assess_repository({
        "name": "lab/quantization-routines",
        "description": "A new quantization method for digital communications.",
    })
    assert result["selection_status"] == "review"
    assert "method-tied-novelty-claim" not in result["selection_signals"]
    assert assess_repository({
        "name": "sfilip/fquantizer",
        "description": "Euclidean lattice-based quantization routines for digital filter design",
    })["selection_status"] == "exclude"


def test_short_rag_application_course_is_excluded() -> None:
    result = assess_repository({
        "name": "rlaalstn1504/2-Day-RAG-Based-AI-Application-Design-2026",
        "description": "Two-day RAG-based AI application design course.",
    })
    assert result["selection_status"] == "exclude"


def test_course_or_workflow_with_explicit_new_method_goes_to_review() -> None:
    for row in (
        {
            "name": "lab/course-project",
            "description": "A course project where we propose a novel Mamba architecture.",
        },
        {
            "name": "lab/causal-workflow",
            "description": "A workflow that introduces a novel causal inference method for treatment effects in machine learning.",
        },
    ):
        result = assess_repository(row)
        assert result["selection_status"] == "review"
        assert result["selection_reason"] == "course-or-utility-with-novel-method-cue"


def test_novel_scientific_ml_methods_remain_includable() -> None:
    rows = (
        "We propose a new protein folding model using deep learning.",
        "We introduce a novel molecular docking method with a graph neural network.",
        "We propose a new robot control policy using reinforcement learning.",
    )
    for description in rows:
        assert assess_repository({"name": "lab/research", "description": description})["selection_status"] == "include"


def test_topics_can_supply_method_context_but_not_novelty_claims() -> None:
    rows = (
        {
            "name": "ciddwd/overlay-translator",
            "description": "A desktop overlay for translating visual novels.",
            "topics": ["visual-novel", "llama-cpp"],
        },
        {
            "name": "Yzw202011/OmniSpace",
            "description": "Tools for creative writing and generative applications.",
            "topics": ["novel-writing", "RAG", "diffusion"],
        },
    )
    for row in rows:
        result = assess_repository(row)
        assert result["selection_status"] == "review"
        assert "method-tied-novelty-claim" not in result["selection_signals"]


def test_survey_that_proposes_a_taxonomy_is_excluded() -> None:
    result = assess_repository({
        "name": "hymie122/RAG-Survey",
        "description": "A survey collecting awesome papers on RAG and proposing a taxonomy.",
        "topics": ["retrieval-augmented-generation"],
    })
    assert result["selection_status"] == "exclude"
    assert result["selection_reason"] == "survey-or-paper-list-repository"


def test_description_led_tutorial_and_lecture_slides_are_excluded() -> None:
    rows = (
        {
            "name": "NVIDIA-AI-IOT/jetson-intro-to-distillation",
            "description": "A tutorial introducing knowledge distillation as an optimization technique for deployment on NVIDIA Jetson",
        },
        {
            "name": "mkang315/CST-YOLO",
            "description": "Lecture presentation slides explaining object detection and YOLO.",
        },
    )
    for row in rows:
        assert assess_repository(row)["selection_status"] == "exclude"


def test_existing_lime_implementation_goes_to_review() -> None:
    result = assess_repository({
        "name": "Ily17as/LIME",
        "description": "An implementation of LIME for explaining machine learning predictions.",
    })
    assert result["selection_status"] == "review"
    assert "method-tied-novelty-claim" not in result["selection_signals"]


def test_orphan_propose_is_not_a_novelty_claim() -> None:
    result = assess_repository({
        "name": "thangquyeenf/research_impl",
        "description": "Implementation my research propose in preference-based reinforcement learning (action preference)",
    })
    assert result["selection_status"] == "review"
    assert "method-tied-novelty-claim" not in result["selection_signals"]


def test_temporal_fusion_transformer_overview_stays_in_review() -> None:
    result = assess_repository({
        "name": "Kostee/tft-overview",
        "description": "Some codes and reflections related to Google's Temporal Fusion Transformer, proposed in 2020",
    })
    assert result["selection_status"] == "review"
    assert result["selection_reason"] == "overview-reproduction-or-dataset"


def test_hpo_rl_paper_result_reproduction_stays_in_review() -> None:
    result = assess_repository({
        "name": "automl/HPO_for_RL",
        "description": "Code of reproducing the results of a paper on hyperparameter optimization for reinforcement learning.",
    })
    assert result["selection_status"] == "review"
    assert result["selection_reason"] == "overview-reproduction-or-dataset"


def test_novel_dataset_is_not_included_as_a_method_contribution() -> None:
    result = assess_repository({
        "name": "tsinghua/visual-tactile-dataset",
        "description": "A novel visual-tactile dataset for robotic manipulation.",
    })
    assert result["selection_status"] == "review"
    assert result["selection_reason"] == "overview-reproduction-or-dataset"


def test_tutorial_is_excluded_even_when_it_mentions_novelty() -> None:
    result = assess_repository({
        "name": "lab/tutorial",
        "description": "A tutorial on a novel Mamba architecture.",
    })
    assert result["selection_status"] == "exclude"


def test_pinned_description_precision_regressions() -> None:
    rows = (
        {
            "name": "NVIDIA-AI-IOT/jetson-intro-to-distillation",
            "description": "A tutorial introducing knowledge distillation as an optimization technique for deployment on NVIDIA Jetson",
        },
        {
            "name": "thangquyeenf/research_impl",
            "description": "Implementation my research propose in preference-based reinforcement learning (action preference)",
        },
        {
            "name": "Kostee/tft-overview",
            "description": "Some codes and reflections related to Google's Temporal Fusion Transformer, proposed in 2020",
        },
        {
            "name": "Paul-Gy/ETH-Deep-Regime-Modeling",
            "description": "A novel application of Google's Temporal Fusion Transformer for financial regime modeling.",
        },
        {
            "name": "Adam-maz/GenAI-assisted-tool-for-Virtual-Screening",
            "description": "Introduces a proof-of-concept toolkit combining existing tools for virtual screening.",
        },
        {
            "name": "sivadst/Genesis",
            "description": "Generate novel molecules and candidates for drug discovery.",
        },
        {
            "name": "suvomx1999/De_Novo_Drug-Generator",
            "description": "Generate novel drug candidates using existing language models.",
        },
        {
            "name": "Ily17as/Interpretable-ML",
            "description": "We propose implementing LIME with XGBoost for interpretable machine learning.",
        },
    )
    for row in rows:
        assert assess_repository(row)["selection_status"] != "include", row["name"]


def test_long_paper_title_compilation_does_not_look_novel() -> None:
    result = assess_repository({
        "name": "USTCPCS/CVPR2018_attention",
        "description": "A compilation of long paper titles and attention-related papers from CVPR 2018.",
    })
    assert result["selection_status"] != "include"


def test_skill_tree_with_new_model_words_is_not_included() -> None:
    result = assess_repository({
        "name": "AlphaSaleAidan/opus-5-5-x-gpt-skill-tree",
        "description": "Opus 5.5 x GPT New Models Skill Tree — 348 preloaded Claude Code + Codex agent skills with a multi-model router: Claude Opus 5.5 leads, Fable writes the markdown handoff, GPT-6 Astra, Sol and Luna build. By Aidan Pierce.",
        "topics": [
            "agent-skills", "agents-md", "ai-agents", "ai-coding", "claude-code",
            "claude-opus", "gpt-6", "gpt-6-astra", "gpt-6-luna", "gpt-6-sol",
            "llm-routing", "model-routing", "multi-agent", "openai-codex",
            "opus-5-5", "prompt-engineering",
        ],
    })
    assert result["selection_status"] != "include"


def test_radio_lora_protocol_is_not_an_ml_method() -> None:
    result = assess_repository({
        "name": "proveskit/drift-protocol",
        "description": "The DRIFT (Dynamic Routing for Inter-satellite Fault Tolerance) protocol is a novel method of ad hoc mesh networking using LoRa communications.",
    })
    assert result["selection_status"] != "include"
    assert "ml-method-cue" not in result["selection_signals"]


def test_lora_with_explicit_ml_context_remains_includable() -> None:
    result = assess_repository({
        "name": "lab/new-lora-method",
        "description": "We propose a new LoRA adapter method for fine-tuning LLMs.",
    })
    assert result["selection_status"] == "include"


def test_backtesting_with_strong_method_evidence_goes_to_review() -> None:
    result = assess_repository({
        "name": "lab/novel-trading-model",
        "description": "We propose a new transformer architecture for backtesting financial time series.",
    })
    assert result["selection_status"] == "review"
    assert result["selection_reason"] == "non-ml-utility-with-ml-contribution-cue"


def test_results_depend_only_on_repository_owned_text_and_fork_flag() -> None:
    base = {"name": "lab/tool", "description": "A transformer implementation", "topics": []}
    noisy = {
        **base,
        "query_ids": ["course.homework"],
        "all_methods": ["transformer"],
        "homepage": "https://paper.example",
        "stargazers_count": 999999,
        "license": "MIT",
    }
    assert assess_repository(base) == assess_repository(noisy)



def test_official_paper_implementations_are_included_without_novel_wording() -> None:
    rows = (
        ("FoundationVision/VAR", "[NeurIPS 2024 Best Paper Award] Official impl. of Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.", ["autoregressive-image-generation"]),
        ("WenjieDu/SAITS", "The official PyTorch implementation of the paper SAITS: Self-Attention-based Imputation for Time Series (arXiv:2202.08516).", ["time-series-imputation"]),
        ("cure-lab/LTSF-Linear", "[AAAI-23 Oral] Official implementation of LTSF-Linear: Are Transformers Effective for Time Series Forecasting?", ["time-series-forecasting"]),
        ("langfengQ/verl-agent", "Official code for the paper Group-in-Group Policy Optimization for Multi-Turn LLM Agents (arXiv:2505.11435).", ["agentic-reinforcement-learning"]),
        ("yujinie98/PatchTST", "Official code for the PatchTST paper on long-term forecasting with time series transformers (ICLR 2023).", ["time-series-forecasting"]),
        ("ali-vilab/VACE", "[ICCV 2025] Official implementations for paper: VACE: All-in-One Video Creation and Editing.", ["video-generation", "video-editing"]),
        ("NExT-GPT/NExT-GPT", "Code and models for ICML 2024 paper, NExT-GPT: Any-to-Any Multimodal Large Language Model.", ["multimodal-large-language-model"]),
    )
    for name, description, topics in rows:
        result = assess_repository({"name": name, "description": description, "topics": topics, "fork": False})
        assert result["selection_status"] == "include", (name, result)
        assert "official-paper-implementation-cue" in result["selection_signals"]


def test_official_paper_route_rejects_weak_or_unrelated_evidence() -> None:
    rows = (
        {"name": "user/bert-reproduction", "description": "Reproduction code for the BERT paper, arXiv:1810.04805."},
        {"name": "student/course-project", "description": "Official implementation of our transformer paper, NeurIPS 2024. Course project."},
        {"name": "lab/survey", "description": "Official code for our survey paper on diffusion, NeurIPS 2024."},
        {"name": "lab/zipline", "description": "Official code for our transformer paper, NeurIPS 2024.", "fork": True},
        {"name": "octocat/octocat", "description": "A machine learning project."},
        {"name": "lab/lora", "description": "Official code for our paper on LoRa networking, NeurIPS 2024."},
        {"name": "lab/gpt-skill-tree", "description": "Official implementation for our Codex skill tree paper about GPT, NeurIPS 2024."},
        {"name": "lab/misc", "description": "Official code for our unrelated image compression paper, NeurIPS 2024."},
        {"name": "lab/concatenated", "description": "Official code for our paper on data management. Separately, discusses transformers. NeurIPS 2024."},
    )
    for row in rows:
        assert assess_repository(row)["selection_status"] != "include", row["name"]


def test_owner_profile_can_pass_only_with_official_research_evidence() -> None:
    result = assess_repository({"name": "lab/lab", "description": "Official implementation of our PatchTST paper on time series forecasting (ICLR 2023)."})
    assert result["selection_status"] == "include"
    assert assess_repository({"name": "lab/lab", "description": "Machine learning"})["selection_reason"] == "owner-profile-repository"


def test_dataset_topic_does_not_block_original_model_implementation() -> None:
    result = assess_repository({
        "name": "lab/new-mamba-model",
        "description": "We propose a novel Mamba architecture for time series forecasting.",
        "topics": ["datasets", "transformer"],
    })
    assert result["selection_status"] == "include"


def test_readme_evidence_promotes_sparse_draggan_and_gpt2_metadata() -> None:
    evidence = ["ml-method-context", "method-contribution", "paper-code-relationship"]
    for name in ("lab/DragGAN", "lab/gpt-2"):
        result = assess_repository({
            "name": name, "description": "Research implementation.",
            "readme_status": "ok", "readme_signals": evidence,
        })
        assert result["selection_status"] == "include", (name, result)
        assert result["selection_reason"] == "readme-supported-paper-method-implementation"
        assert set(evidence) <= set(result["selection_signals"])


def test_readme_promotion_requires_active_status_and_related_signals() -> None:
    qualifying = ["ml-method-context", "method-contribution", "paper-code-relationship"]
    base = {"name": "lab/project", "description": "A useful machine learning project."}
    assert assess_repository({**base, "readme_status": "missing", "readme_signals": qualifying})["selection_status"] == "review"
    unrelated = ["ml-method-context", "method-contribution", "paper-reference"]
    result = assess_repository({**base, "readme_status": "ok", "readme_signals": unrelated})
    assert result["selection_status"] == "review"
    assert "paper-reference" in result["selection_signals"]


def test_readme_negative_cues_and_zipline_prevent_promotion() -> None:
    positives = ["ml-method-context", "method-contribution", "paper-code-relationship"]
    for negative in ("course-cue", "reproduction-cue", "dataset-only-cue"):
        result = assess_repository({
            "name": "lab/project", "description": "A machine learning project.",
            "readme_status": "unchanged", "readme_signals": [*positives, negative],
        })
        assert result["selection_status"] == "review", (negative, result)
    zipline = assess_repository({
        "name": "quantopian/zipline",
        "description": "Zipline, a Pythonic Algorithmic Trading Library",
        "topics": ["algorithmic-trading", "python", "quant", "zipline"],
        "fork": False, "readme_status": "ok", "readme_signals": positives,
    })
    assert zipline["selection_status"] == "exclude"
    assert zipline["selection_reason"] == "non-ml-utility"


def test_metadata_readme_synergy_promotes_official_paper_implementations() -> None:
    cases = (
        (
            {
                "name": "XingangPan/DragGAN",
                "description": "Official Code for DragGAN (SIGGRAPH 2023)",
                "topics": ["artificial-intelligence", "generative-adversarial-network", "generative-models"],
            },
            ["paper-reference", "ml-method-context"],
        ),
        (
            {
                "name": "openai/gpt-2",
                "description": 'Code for the paper "Language Models are Unsupervised Multitask Learners"',
                "topics": ["paper"],
            },
            ["paper-reference", "ml-method-context", "paper-code-relationship"],
        ),
        (
            {"name": "guoyww/AnimateDiff", "description": "Official implementation of AnimateDiff."},
            ["paper-reference", "ml-method-context", "method-contribution"],
        ),
    )
    for metadata, evidence in cases:
        result = assess_repository({
            **metadata, "readme_status": "ok", "readme_signals": evidence,
        })
        assert result["selection_status"] == "include", (metadata["name"], result)
        assert result["selection_reason"] == "readme-supported-paper-method-implementation"


def test_metadata_readme_synergy_rejects_generic_apps_and_unrelated_mentions() -> None:
    rows = (
        {
            "name": "lab/official-llm-cli",
            "description": "Official implementation of a command line application for LLM users.",
            "readme_signals": ["paper-reference", "ml-method-context"],
        },
        {
            "name": "lab/gpt-dashboard",
            "description": "A dashboard with a citation to a transformer paper.",
            "topics": ["transformer"],
            "readme_signals": ["paper-reference", "ml-method-context"],
        },
    )
    for row in rows:
        result = assess_repository({**row, "readme_status": "ok"})
        assert result["selection_status"] == "review", (row["name"], result)


def test_official_codebase_with_readme_paper_code_relation_promotes_without_novelty_wording() -> None:
    pinned = {
        "name": "facebookresearch/ijepa",
        "description": (
            "Official codebase for I-JEPA, the Image-based Joint-Embedding Predictive Architecture. "
            'First outlined in the CVPR paper, "Self-supervised learning from images with a '
            'joint-embedding predictive architecture."'
        ),
        "topics": [],
        "fork": False,
    }
    signals = ["paper-reference", "ml-method-context", "paper-code-relationship"]
    assert assess_repository(pinned)["selection_status"] == "review"
    result = assess_repository({
        **pinned, "readme_status": "ok", "readme_signals": signals,
    })
    assert result["selection_status"] == "include"
    assert result["selection_reason"] == "readme-supported-paper-method-implementation"


def test_official_applied_codebase_without_local_readme_paper_relation_stays_review() -> None:
    result = assess_repository({
        "name": "lab/official-dashboard",
        "description": "Official PyTorch codebase for an operations dashboard, with a transformer paper citation (CVPR 2024).",
        "topics": ["transformer", "dashboard"],
        "readme_status": "ok",
        "readme_signals": ["paper-reference", "ml-method-context"],
    })
    assert result["selection_status"] == "review"
