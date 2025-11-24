# Jenga-AI: Project Analysis, Roadmap, and Issues

This document provides a comprehensive architectural analysis, a strategic roadmap, and a detailed list of actionable issues for the Jenga-AI project. It is designed to be a living document to guide development, from stabilizing the foundation to future growth.

## 1. Executive Summary & System Analysis

### 1.1. Vision
Jenga-AI aims to democratize advanced, context-aware NLP for African languages by providing a framework that simplifies the creation of powerful multi-task models. The core mission is to lower technical and financial barriers for local developers to solve relevant regional problems.

### 1.2. Architectural Strengths
- **Modularity:** Excellent separation of concerns (`core`, `data`, `tasks`, `training`).
- **Config-Driven:** Reproducible and clear experiment definition via YAML and `dataclasses`.
- **Industry Standards:** Built on the solid foundation of PyTorch and Hugging Face.
- **Innovative Core:** The `AttentionFusion` layer provides a unique, research-informed advantage.

### 1.3. Key Risks and Bottlenecks
1.  **Deployment Gap:** The framework lacks a path from a trained model to a usable inference endpoint. **This is the highest priority bottleneck.**
2.  **Testing & CI:** The absence of a robust CI pipeline and comprehensive test coverage puts project stability at risk.
3.  **Documentation Mismatch:** Documentation describes features that are not yet implemented, which can mislead users.
4.  **Inflexible Data Handling:** The `DataProcessor` is tightly coupled to file paths, limiting its use in diverse production environments.

---

## 2. Strategic Roadmap: Milestones

The development is broken down into five key milestones:

1.  **Milestone 1: Foundational Stability & Developer Experience:** Solidify the core, establish automated quality gates, and improve the developer workflow.
2.  **Milestone 2: End-to-End Production Viability:** Bridge the critical gap from training to deployment.
3.  **Milestone 3: Core Module Robustness (Existing Code):** Ensure all existing components are well-tested, documented, and refactored to production standards.
4.  **Milestone 4: Advanced Capabilities & Extensibility:** Enhance the framework’s power and flexibility.
5.  **Milestone 5: Community, Ecosystem, and Capacity Building:** Grow a healthy community through documentation, tutorials, and training.

---

## 3. Detailed Issue Tracker

### Milestone 1: Foundational Stability & Developer Experience

*   #### Issue #1.1: Establish a CI/CD Pipeline for Automated Testing
    *   **Title:** `feat(devops): Implement CI/CD pipeline with GitHub Actions`
    *   **Description:** To ensure code quality and prevent regressions, we need to automate our testing process. This issue involves setting up a GitHub Actions workflow that runs on every pull request and push to the `main` branch.
    *   **User Story:** As a core maintainer, I want all pull requests to be automatically tested, so that we can merge code with confidence and prevent breaking changes.
    *   **Acceptance Criteria:**
        1.  A `.github/workflows/ci.yml` file is created.
        2.  The workflow triggers on PRs targeting `main`.
        3.  The workflow installs all project dependencies from `requirements.txt`.
        4.  The workflow runs the entire `pytest` suite.
        5.  The workflow includes a code linting step (using `ruff`).
        6.  A "passing" status check is required before PRs can be merged.

*   #### Issue #1.2: Synchronize Documentation with Reality
    *   **Title:** `docs(readme): Update documentation to reflect current implementation status`
    *   **Description:** Our documentation currently describes features (like deployment) that are not yet implemented. We need to revise the `README.md` and other docs to clearly distinguish between current features and future roadmap items.
    *   **User Story:** As a new user, I want the documentation to accurately represent the current state of the project, so I don't get confused or frustrated by missing features.
    *   **Acceptance Criteria:**
        1.  Add a "Project Status" section to the `README.md` indicating it's in active development.
        2.  In sections describing unimplemented features, add a note: `Note: This feature is on our roadmap and is not yet implemented.`
        3.  Verify that the Quickstart example in the `README.md` is runnable and correct.

### Milestone 2: End-to-End Production Viability

*   #### Issue #2.1: Implement a Robust Model Export and Inference Pipeline
    *   **Title:** `[EPIC] feat(deployment): Implement end-to-end model export and inference pipeline`
    *   **Description:** This is the most critical bottleneck. We need to implement the `export` and `inference` modules to make trained models usable.
    *   **Sub-Issue #2.1.1: Implement Model Export Functionality**
        *   **User Story:** As a data scientist, I want to save a trained multi-task model to a single, portable artifact, so I can easily load it for inference.
        *   **Acceptance Criteria:**
            1.  The `multitask_bert/deployment/export.py` script is implemented.
            2.  It saves the model's state dict, the full `ExperimentConfig`, and the tokenizer files into a specified directory.
            3.  A unit test is added to verify that a trained model can be exported without errors.
    *   **Sub-Issue #2.1.2: Create a Unified Inference Class**
        *   **User Story:** As a developer, I want to load an exported model artifact and run predictions for any of its supported tasks using a simple, unified interface.
        *   **Acceptance Criteria:**
            1.  `multitask_bert/deployment/inference.py` contains an `InferenceWrapper` class.
            2.  The class is initialized with the path to an exported model directory.
            3.  It provides a `predict(text: str, task_name: str) -> dict` method.
            4.  A unit test is added that exports a dummy model and then runs inference with it.
    *   **Sub-Issue #2.1.3: Build a FastAPI Service for Real-time Inference**
        *   **User Story:** As an MLOps engineer, I want to expose a trained model via a REST API, so that other services can consume its predictions.
        *   **Acceptance Criteria:**
            1.  A new top-level `api/` directory is created with a `main.py` FastAPI application.
            2.  It includes a `/predict` endpoint that accepts `text` and `task_name`.
            3.  A `Dockerfile` is provided to containerize the API.
            4.  Basic API tests are added.

### Milestone 3: Core Module Robustness (Existing Code)

*   #### Issue #3.1: Achieve 90%+ Test Coverage for All Core Modules
    *   **Title:** `test(core): Increase test coverage to >90% for all existing modules`
    *   **Description:** A robust test suite is essential for stability. We must write unit and integration tests for all critical components, including `MultiTaskModel`, `DataProcessor`, `Trainer`, `AttentionFusion`, and all defined `tasks`.
    *   **User Story:** As a developer, I want to be able to refactor a core component and have a comprehensive test suite to verify that I haven't broken anything.
    *   **Acceptance Criteria:**
        1.  Run `pytest --cov=multitask_bert` to generate a coverage report.
        2.  Write new tests in `multitask_bert/tests/` for any untested functions, classes, or edge cases.
        3.  The final test coverage for the `multitask_bert` module must exceed 90%.

*   #### Issue #3.2: Add Comprehensive Docstrings and Type Hinting
    *   **Title:** `docs(code): Add Sphinx-compatible docstrings and full type hinting to all public APIs`
    *   **Description:** To improve code clarity and maintainability, all public classes, methods, and functions across the `multitask_bert` module should have complete type hinting and detailed docstrings.
    *   **User Story:** As a new contributor, I want to understand what a function does, its parameters, and what it returns by reading its docstring, without having to read its source code.
    *   **Acceptance Criteria:**
        1.  All public-facing functions and methods in `.py` files under `multitask_bert/` have complete type hints.
        2.  All public-facing functions, methods, and classes have docstrings explaining their purpose, arguments (`Args:`), and return values (`Returns:`).
        3.  The docstring format is consistent (e.g., Google-style or Sphinx-style).

*   #### Issue #3.3: Refactor `core.config` for Automatic Validation
    *   **Title:** `refactor(core): Implement Pydantic-style validation for ExperimentConfig`
    *   **Description:** Currently, an invalid YAML configuration might only cause an error deep into the training process. We should validate the configuration object upon loading to provide immediate, clear feedback to the user.
    *   **User Story:** As a user, if I provide a bad `data_path` or an invalid `task_type` in my YAML file, I want the program to fail fast with a clear error message.
    *   **Acceptance Criteria:**
        1.  The `dataclasses` in `multitask_bert/core/config.py` are potentially replaced with `pydantic.BaseModel`, or validation logic is added.
        2.  Validation checks are implemented for: file paths existing, valid vocabulary for `task_type`, etc.
        3.  Unit tests are added to ensure both valid and invalid configs are handled correctly.

*   #### Issue #3.4: Create a Task Registry for Extensibility
    *   **Title:** `refactor(tasks): Implement a task registry to replace if/elif logic`
    *   **Description:** Adding new tasks currently requires modifying `if/elif` blocks in the codebase. A registry pattern would allow new tasks to be added declaratively, making the framework more extensible.
    *   **User Story:** As a developer, I want to add a new custom task to the framework without modifying the core training or data processing logic.
    *   **Acceptance Criteria:**
        1.  A `TaskRegistry` class is created in `multitask_bert/core/registry.py`.
        2.  Tasks can be registered with a unique name (e.g., `@TaskRegistry.register("my_custom_task")`).
        3.  The `DataProcessor` and `Trainer` use the registry to look up task classes by their string name from the config.
        4.  All existing tasks are refactored to use the registry.

### Milestone 4: Advanced Capabilities & Extensibility

*   #### Issue #4.1: Refactor DataProcessor for Greater Flexibility
    *   **Title:** `refactor(data): Decouple DataProcessor from file-based sources`
    *   **Description:** The current `DataProcessor` is too rigid. We need to refactor it to support a wider range of data sources.
    *   **User Story:** As a data scientist, I want to train a model using data from a pandas DataFrame or a database, not just from a file.
    *   **Acceptance Criteria:**
        1.  Create a `BaseDataSource` abstract class.
        2.  Implement `FileDataSource` and `DataFrameDataSource` subclasses.
        3.  Refactor `DataProcessor` to accept a list of `BaseDataSource` objects.
        4.  Update examples to show usage with both a file and a DataFrame.

*   #### Issue #4.2: Explore and Implement True Simultaneous Multi-Tasking
    *   **Title:** `feat(core): Implement optional strategy for simultaneous multi-task training`
    *   **Description:** To improve inter-task learning, we should offer an alternative training strategy that processes batches containing mixed-task data. This is a research-heavy task.
    *   **User Story:** As an NLP researcher, I want to experiment with a simultaneous multi-tasking strategy to see if it improves model performance on related tasks.
    *   **Acceptance Criteria:**
        1.  **Research Spike:** Create a design document (`docs/design/simultaneous_mtl.md`) outlining the proposed architecture.
        2.  Implement the new strategy as an optional mode in the `Trainer`, configurable via YAML.
        3.  Add an example experiment and tests for the new strategy.

### Milestone 5: Community, Ecosystem, and Capacity Building

*   #### Issue #5.1: Create a Comprehensive Tutorial Series
    *   **Title:** `docs(tutorial): Develop end-to-end tutorials for common use cases`
    *   **Description:** We need tutorials that go beyond the quickstart and guide users through a full project lifecycle.
    *   **Acceptance Criteria:**
        1.  Create a tutorial for "Building a Swahili Sentiment and NER Model," covering data prep, config, training, and deployment with the FastAPI service.
        2.  Create a developer tutorial on "How to Add a New Task to Jenga-AI."
        3.  Ensure all tutorials are tested and runnable.

*   #### Issue #5.2: Plan and Announce a Contributor Workshop Series
    *   **Title:** `project(community): Plan and schedule a series of capacity-building workshops`
    *   **Description:** To empower our team and the wider community, we will host a series of virtual workshops. This task is to create the curriculum and schedule.
    *   **Sub-Issue #5.2.1: Curriculum for "Jenga-AI Contributor Kickstart"**
        *   **Goal:** Onboard new developers.
        *   **Topics:** Project vision, architecture overview, dev environment setup, running tests, and tackling a "good first issue."
    *   **Sub-Issue #5.2.2: Curriculum for "Advanced NLP with Jenga-AI"**
        *   **Goal:** Upskill developers.
        *   **Topics:** Deep dive into attention fusion, adding new tasks, and interpreting model outputs.
    *   **Sub-Issue #5.2.3: Curriculum for "Deploying Jenga-AI at Scale"**
        *   **Goal:** Enable production use.
        *   **Topics:** Using the FastAPI service, containerization with Docker, and MLOps best practices.
