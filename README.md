# Omics-Embedding-Learning

Codes for CE7412 project (Spring 2024)

## Datasets & Integrated datasets

- All datasets and integrated datasets used in this project can be found [here](https://drive.google.com/drive/folders/1ZZNR2inwiXpbOdRIWa6YJDssglFGXx6c?usp=sharing).

- Default folder tree setting for this project

```bash
├── datasets                    # original datasets
│   ├── COAD
│   ├── ESCA         
│   ├── LIHC
│   ├── OV
│   ├── STAD
│   └── UCEC
├── Integrated_datasets_128     # integrated datasets using GAN, selected features=128
│   └ ...                
├── Integrated_datasets_256     # integrated datasets using GAN, selected features=256
│   └ ...                  
├── Integrated_datasets_MLP     # integrated datasets using MLP, selected features=128
│   └ ...                  
└── Omics-Embedding-Learning
    └ ...                       # codes of this project
```