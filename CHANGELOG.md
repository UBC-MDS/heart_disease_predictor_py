# CHANGELOG ✨

This document outlines the improvements made to the data analysis project based on feedback received from the DSCI 522 teaching team and peer reviews. Each section lists the feedback, the changes made to address it, and where to find evidence of the improvements in the project repository.

## 1. Feedback on Past Milestones from the Teaching Team 🚀

### Milestone 1 - 1. Ensure Proper Version Pinning for All Packages in environment.yml

**Feedback:**  

Programming language and/or package versions are pinned using >= instead of =. This means that each time the environment is built in the future, the most recent version of the programming language and/or package will be installed in the environment. This will lead to the environment not being able to be reproducibly built in the future.

Beautiful work overall in your submission for Milestone 1! However, I deducted points because (1) version number was not listed for some package dependencies listed under environment.yml (e.g., ipykernel, matplotlib are missing the version number). Please list the exact version for all dependencies.
(2) When listing the version of the dependency, please use “=“ instead of “>=“ (e.g., “scikit-learn>=1.5.1” should be “scikit-learn=1.5.1”). Using >= allows Conda to install any version greater than or equal to 1.5.1, potentially introducing changes that affect model performance or metrics. Using = instead of >= in an environment.yml file is critical for reproducibility, it ensures that the exact same package versions are installed every time the environment is created.

**Changes Made:**

- Specify the versions in the environment.yml.
- Replace all the ">=" with "=" for reproducibility.
  
**Evidence:**

- [Commit #cab69a5](https://github.com/UBC-MDS/heart_disease_predictor_py/commit/cab69a5b357f608a35631c4af88a4d158cc288ae): Specify the versions in the environment.yml.

---

### Milestone 1 - 2. No software license was specified

**Feedback:**  
Creative commons license was listed for project report but MIT license is missing for the project code

**Changes Made:**

- Added the MIT license to the project code in the LICENSE file, and the Creative Commons license remains for the project report.
  
**Evidence:**

- [LICENSE](https://github.com/UBC-MDS/heart_disease_predictor_py/blob/f16e4c6f8b0c350b3f49321834d8cfbafab516fe/LICENSE#L127): Update license with MIT for software code with reference to Tiffany's note.

---

### Milestone 1 - 3. Reproducibility: fix environment name in README.md to activate environment correctly

**Feedback:**  
Usage documentation could be improved for clarity (i.e., it is not explicitly clear to the user how to use the project, or some of the wording is confusing, some guessing and/or trial and error had to be performed to run the project).

I was able to run the analysis after some trial and error, and reproduce your work! However, I deducted points because (1) if the user set up virtual environments in the way as instructed using environment.yml file, they would not be able to activate the conda environment. This is because in the environment.yml the environment name is "heart_disease_predictor", but the user were instructed to run activate another environment with a slightly different name called "conda activate heart_disease_env" (2) the instructions for setting up environment using conda-lock is wrong. Here is the error that I got "$ conda-lock install --name heart_disease_env --file conda-lock.yml
Usage: conda-lock install [OPTIONS] [LOCK_FILE]
Try 'conda-lock install --help' for help.

Error: No such option: --file Did you mean --auth-file?". (3) the instructions about opening jupyter lab is redundant, can cause confusing for users new to jupyter lab (users were instructed to run ```bash jupyter lab ```, then instructed to run ```bash jupyter lab src/heart_disease_predictor_report.ipynb ```, but the user would not be able to run the second command when the first jupyter lab is still running).

**Changes Made:**

- Changed the conda activate command from "conda activate heart_disease_env" to "conda activate heart_disease_predictor".
  
**Evidence:**

- [Commit #13074c4](https://github.com/UBC-MDS/heart_disease_predictor_py/commit/13074c45c29a48f2e023974bb7e2f422ebaefe3b#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5R76): Line 76 with the correct conda activate command.

---

### Milestone 2 - 1. Missing platform key and value from the docker-compose.yml file

**Feedback:**  
The platform key and value is missing from the docker-compose.yml file, causing issues when running on different chip architectures.

Overall the docker-compose file has the essential components. However, I deducted points because your file is missing "platform" specification. It's important to specify platform like: linux/amd64, because if the Docker image architecture differs from the host system (e.g., Apple Silicon), you will run into non-compatibility issues

**Changes Made:**

- Specify the platform key to `linux/amd64` in `docker-compose.yml`.
  
**Evidence:**  

- [docker-compose.yml](https://github.com/UBC-MDS/heart_disease_predictor_py/blob/8070d1b28ef6e3a8fa822922e807647a5344a7cb/docker-compose.yml#L12): Line 12 with the specified platfrom.

---

### Milestone 2 - 2. Reproducibility

**Feedback:**  
The reproducibility is great! I was able to docker compose up your docker image. However, I deducted a point for (1) in README file, you instructed the users to regenerate conda-lock file using ``` conda-lock install --name heart_disease_predictor --file environment.yml``` this is the wrong command line. This code is used to install a conda environment. (2) it's redundant to have the "Using the Docker Container" section asking the user to pull and mount docker image manually, because you already have the docker compose file. users should be instructed to directly do "docker compose up". It's also not good practice to use the image tag "latest" because it does not guarantee the latest is actually pulled. Please use specific tags instead, like how you did it in docker compose file. (3) it's redundant to ask user to create a docker-compose file by copying your docker compose file in your readme, as you already have it in your root directory. (4) you probably don't need the version: '3' in your docker compose file (5) you don't need to ask them to conda deactivate the environment if they are using docker (6) you do need to tell the user how to exit docker image that is running though: "Press Ctrl + C to exit the container"

**Changes Made:**  

1. Delete the instruction in README for regenerating conda-lock file to install a conda environment.
2. Delete the instruction in README for manually pulling and mounting docker image, keeping a simple instruction to use `docker compose up`.
3. Delete the instruction in README for creating a docker-compose.yml by copying the file in README.
4. Delete the version '3' in `docker-compose.yml`.
5. Add instruction of how to exit docker image by using Ctrl + C.
  
**Evidence:**  

- [README.md](https://github.com/UBC-MDS/heart_disease_predictor_py/blob/main/README.md): Changes 1, 2, 3, 5 in `README.md`.
- [Commit #8070d1b](https://github.com/UBC-MDS/heart_disease_predictor_py/commit/8070d1b28ef6e3a8fa822922e807647a5344a7cb#diff-e45e45baeda1c1e73482975a664062aa56f20c03dd9d64a827aba57775bed0d3L1): Change 4 in `docker-compose.yml`.

---

## 2. [Feedback from Peer Review](https://github.com/UBC-MDS/data-analysis-review-2024/issues/14) 🔧

### 2.1 Missing Cloning Instructions

**Feedback 1 from [Yasmin Hassan](https://github.com/yasmin2424):**  
Your instructions for running the analysis lack a critical step. You need to include instructions on how to clone your repository to ensure users can properly access and run your work.

**Response and Changes Made:**

- Added the missing cloning instructions in the `README.md`.
- [Response to Yasmin's Feedback](https://github.com/UBC-MDS/data-analysis-review-2024/issues/14#issuecomment-2543367111)
  
**Evidence:**

- [Commit #2822eb1](https://github.com/UBC-MDS/heart_disease_predictor_py/commit/2822eb1dfb8b6f17a59ac3b79d608665f87f69d5): Addressed the missing cloning instructions in the `README.md` by adding a `Clone the Repository` section.
- [Response to Yasmin's Feedback](https://github.com/UBC-MDS/data-analysis-review-2024/issues/14#issuecomment-2543367111)

---

### 2.2 Script Errors and Docker Configuration Issue

**Feedback 2 from [Yasmin Hassan](https://github.com/yasmin2424):**  
When running the scripts directly, they fail with errors like:
python: can't open file '/home/jovyan/scripts/download_data.py': [Errno 2] No such file or directory
This happens because your docker-compose.yml file points to an incorrect working directory. I recommend editing the file to ensure it always points to .:/home/jovyan/.

**Clarification and Changes Made:**

- Updated the `volumes` in `docker-compose.yml` from `/home/jovyan/work` to `/home/jovyan`.
- [Clarification](https://github.com/UBC-MDS/data-analysis-review-2024/issues/14#issuecomment-2543367111) on why the scripts failed to run due to the user being in the wrong directory.
  
**Evidence:**

- [Commit #86c96ba](https://github.com/UBC-MDS/heart_disease_predictor_py/commit/86c96ba813f13ce49fd58eff172f60196c688864#diff-e45e45baeda1c1e73482975a664062aa56f20c03dd9d64a827aba57775bed0d3L8-R8): Changed the `volumes` in `docker-compose.yml` to `/home/jovyan`..

---

### 2.3 Error Logging

**Feedback 3 from [Yasmin Hassan](https://github.com/yasmin2424):**  
Instead of printing errors directly to the console, consider outputting them to a log file (e.g., validation_error.log). Printing errors as output may lead to them being overlooked, but logging them ensures they are properly documented and can be reviewed as needed.

**Response and Changes Made:**

- Add a new `setup_logger.py` script to implement logging across all our scripts. Now, all errors will be logged into a single ERROR.log file.
- [Response to Yasmin's Feedback](https://github.com/UBC-MDS/data-analysis-review-2024/issues/14#issuecomment-2543367111)
  
**Evidence:**

- [Commit #8c8e781](https://github.com/UBC-MDS/heart_disease_predictor_py/commit/8c8e7811a15384079661f93d7a34beedede7dcd7): Add a logger function and log all errors during running the scripts.

---

## Conclusion📝

This changelog summarizes all changes made to the project based on the feedback received. Each section is linked to the specific commits or file paths that demonstrate the improvements.
