[![pipeline status](https://gitlab.com/aferdina/MultiArmedBandits/badges/main/pipeline.svg)](https://gitlab.com/aferdina/MultiArmedBandits/-/commits/main)
[![coverage report](https://gitlab.com/aferdina/MultiArmedBandits/badges/main/coverage.svg)](https://gitlab.com/aferdina/MultiArmedBandits/-/commits/main)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Latest Release](https://gitlab.com/aferdina/MultiArmedBandits/-/badges/release.svg)](https://gitlab.com/aferdina/MultiArmedBandits/-/releases)

# Multiarmed Bandits<!-- omit in toc -->

- [Main Features](#main-features)
- [Description](#description)
  - [Overview](#overview)
- [Installation](#installation)
  - [First Steps](#first-steps)
- [Usage](#usage)
- [Tests](#tests)
  - [Git pre-commits](#git-pre-commits)
- [Contributing](#contributing)
- [License](#license)

## Main Features

| **Features**                | **Multiarmedbandits** |
| --------------------------- | ----------------------|
| Documentation               | :heavy_check_mark: |
| Custom environments         | :heavy_check_mark: |
| Custom policies             | :heavy_check_mark: |
| Common interface            | :heavy_check_mark: |
| Ipython / Notebook friendly | :heavy_check_mark: |
| PEP8 code style             | :heavy_check_mark: |
| High code coverage          | :heavy_check_mark: |
| Type hints                  | :heavy_check_mark: |

## Description

### Overview

This package contains

1. functionalities and modules to write algorithms for multi-armed bandits,
2. write environments for multi-armed bandits,
3. test algorithms and compare them with each other.

In addition, some experiments are provided to analyse existing algorithms.

## Installation

### First Steps

Run

```sh
poetry install
```

in your working directory.

## Usage

## Tests

### Git pre-commits

Git pre-commits are a type of hook in Git that allows you to run custom scripts or commands automatically before making a commit. Pre-commit hooks provide a way to enforce certain checks, validations, or actions before allowing a commit to be made, helping to maintain code quality and consistency in a project.

When you run git commit, Git triggers the pre-commit hook, which executes the defined script or command. The hook script can perform various tasks, such as running tests, code linting, formatting checks, static analysis, or any other custom validations specific to your project's requirements.

The pre-commit hook can examine the changes being committed, access the staged files, and even reject the commit if the script fails or returns a non-zero exit code. This allows you to enforce rules or standards on the committed code and prevent potentially problematic changes from being added to the repository.

1. Create a git pre-commit hook by creating a pre-commit bash script by running the following lines of code inside your repository:

    ```sh
    touch .git/hooks/pre-commit
    ```

2. Then add code to the bash script by running:

    ```sh
    open .git/hooks/pre-commit
    ```

3. Add the following code to the file

    ```bash
    #!/bin/zsh

    # Activate the virtual environment
    source .venv/bin/activate

    echo "Running Pytest..."
    pytest tests/

    if [ $? -ne 0 ]; then
      echo "Pytest failed. Aborting commit."
      exit 1
    fi
    echo "Check Codestyle..."
    make check-codestyle

    if [ $? -ne 0 ]; then
      echo "Codestyle check failed. Aborting commit."
      exit 1
    fi
    ```

4. The last step is to make the file executable by running:

    ```sh
    chmod +x .git/hooks/pre-commit
    ```

Now when a git commit is started, pytest is run beforehand. If a test is not successful, the commit is aborted.  

## Contributing

## License
