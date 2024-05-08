## A Toy Example of IP Agents Rating System
The framework is based on Google Deepmind's [Concordia](https://github.com/google-deepmind/concordia), which is a pipeline for configuring an environment and interactive agents within it, based on LLMs. Under this framework, all kinds of scenarios are theoretically possible. It is also possible to introduce human interactors and real-world information into the system.
For our system, we additionally collect the generated data of scenarios and agents' responses, let human users rate the agents' responses, and use the ratings to train the agents' models.  

### Installation
1.  Clone value_:

    ```shell
    git clone https://github.com/fringsoo/valuia.git 
    cd valuia
    ```

2.  Install Concordia:

    ```shell
    pip install --editable .[dev]
    ```
3.  Download Meta-Llama-3-8B-Instruct and sentence-transformers:

    ```shell
    git lfs install
    git clone https://hf-mirror.com/meta-llama/Meta-Llama-3-8B-Instruct
    git clone https://hf-mirror.com/sentence-transformers/all-mpnet-base-v2
    ```

### Run the toy example
Try out ip_figure.ipynb

### Data Format
The collected data is stored in the "interaction_database" folder in CSV format.  
For each IP figure, we build a CSV file, with two columns, i.e. the scenarios and the responses. The same scenario may correspond to multiple responses.

### Training
TBD.