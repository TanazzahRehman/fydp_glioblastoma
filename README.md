# ADVANCING RADIONGENOMICS: AI ENABLED GLIOBLASTOMA SUBTYPE PREDICTION

# Glioblastoma Research Project

This repository contains files and resources related to our research project on glioblastoma.

## About Glioblastoma

Glioblastoma, also known as glioblastoma multiforme (GBM), is one of the most aggressive and lethal types of brain cancer. It originates in the brain's glial cells, which support and protect neurons. Glioblastoma is characterized by its rapid growth and tendency to infiltrate surrounding brain tissue, making it difficult to treat. Standard treatment includes surgery, radiation therapy, and chemotherapy, but the prognosis remains poor with a median survival time of approximately 15 months after diagnosis. Ongoing research is focused on understanding the molecular mechanisms of glioblastoma, developing novel therapeutic approaches, and improving patient outcomes.


## Getting Started

To get started with this project, clone the repository and install the required dependencies.

```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### Prerequisites

Make sure you have the following installed:
- Python 3.8+
- Git
- Git LFS (for handling large files)
- NVIDIA GPU 
- 16GB RAM
- CUDA 11.8+

### Installation

Create a virtual environment and install the dependencies:

```sh
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

## Usage

To run the data processing scripts:

```sh
python scripts/data_processing.py
```

To train the model:

```sh
python scripts/main.ipynb
```

## Contributing

We welcome contributions to this project. Please open an issue or submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
