# bachelor-project
VU-CS 2024 Bachelor Thesis Project - Formal Methods for AI Models in Healthcare

# VU-CS 2024 Bachelor Project Artifact for Thesis Project
## Formal Methods for AI Models in Healthcare

This repository contains the artifact for the bachelor thesis exploring the use formal methods for AI Models in the healthcare sector. THis section aims to serve as a comprehensive guide for installing, utilizing and evaluating the artifact related to this thesis. The main cause was to explore the implementation of formal methods for five AI models, which were trained on a heart disease dataset. This aims to determine the usability of formal methods in development for AI models throughout the healthcare sector. The utilized dataset was obtained from the UCI repository and contains both categorical and numerical features, such as age and cholesterol levels, aiming to predict the presence of heart disease in a target population. For this project, seven formal properties were tested for each AI model:

1. Correctness
2. Safety
3. Robustness
4. Fairness
5. Temporal Consistency
6. Generalizability
7. Scalability

This section provides the necessary guidelines for recreating the experimental setup, accessing the code for this implementation and understanding the specifications required. The artifact comprises of five Python files, each corresponding to one of the five implemented AI models; a decision tree model, a random forest model, a logistic regression model, a k-nearest-neighbors model and a support vector machine model. After executing, each model is trained on the dataset and outputs its accuracy to the user. The user is then prompted to choose between nine options, eight of which verify a formal property and output the result, while the ninth option exits the program.

## Description and Requirements
To recreate the experiment specific software specifications are recommended. In this iteration of the experiment, the following environment was utilized:

- Programming Language: Python 3.12.4
- IDE: Visual Studio Code
- Operating System: 64-bit version of Windows 11

While the artifact likely executes as expected on different operating systems, it was only tested on the above system specifications. Furthermore, in order for the implemented AI models and formal methods to compile, the following python libraries need to be installed on the executing system:

1. 'torch', 'pandas', 'numpy' - for manipulating data and as dependencies for the following libraries.
2. 'sklearn' for the implementation and training of the five AI models.
3. 'z3-solver' and 'art' for verifying correctness, safety and robustness.
4. 'time' and 'psutil' for tracking the scalability of models.

### Security, privacy and ethical concerns
The process of executing the artifact involves minimal security risks, as it functions locally. However, the user should ensure that all data follows ethical guidelines for privacy. If the original dataset is used, it should be obtained directly from the UCI dataset repository. Finally, as this artifact does not directly access external networks, potential security threats are mitigated.

## Hardware dependencies
This artifact was only tested and evaluated using one constant system. While the specifications of this system are not the minimum requirements, having less available computational resources can lead to increased computation times. Especially the verification of formal properties like robustness, which perform a sequence of adversarial attacks using the ART library are computationally intensive and executing them on weaker systems may lead to time-out errors. The relevant hardware specifications of the tested systems are:

- Processor: 16-core Intel CPU with 3.4GHz
- Memory: 32GB DDR4

## Set-Up
The installation of this artifact should ensue after verifying that the required software and hardware specifications are met. The relevant python libraries can be installed by executing the Python command 'pip install torch pandas numpy sklearn z3-solver art time psutil' locally. Furthermore, after locally storing the heart disease dataset from the UCI repository, it is necessary to include the path variable for each python file (the path variable is defined at line 35 in each file).

### Basic Test
After installation, basic functionality can be tested by compiling any one of the five Python files. If the output displays correctly with no encountered errors, the user should see the initial accuracy of the model and be prompted with a choice of nine options. This confirms the model's basic functionality. The user can then proceed to call formal methods for further evaluation of the model by inputting an integer from 1 to 8, depending on the desired method.

## Notes on Reusability
The implementation of the project is designed with flexibility as a focal point and can be adapted for different datasets with similar features. Users can utilize a new dataset by updating the path variable in each of the five files. The achieved results of this implementation can vary depending on the dataset, where similar datasets with both categorical and numerical features should perform more consistently. 

For more details, please refer to the provided comments within the code.
