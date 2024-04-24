import pandas as pd

n = 10
file_path1 = 'GDC/GDC-PANCAN.htseq_fpkm-uq.tsv'
file_path2 = 'GDC/GDC-PANCAN.methylation27.tsv'
file_path3 = 'GDC/GDC-PANCAN.mirna.tsv'

file_path4 = 'dataset/COAD/adjacency.csv'
file_path5 = 'dataset/COAD/label.csv'
file_path6 = 'dataset/COAD/miRNA.csv'
file_path7 = 'dataset/COAD/mRNA.csv'

file_path8 = 'data/A.tsv'
file_path9 = 'data/labels.tsv'
file_path10 = 'anno/B_anno.csv'

gene_expression_data = pd.read_csv(file_path1, sep='\t', nrows=n)
DNA_methylation_data = pd.read_csv(file_path2, sep='\t', nrows=n)
miRNA_expression_data = pd.read_csv(file_path3, sep='\t', nrows=n)

adjacency_data = pd.read_csv(file_path4, sep=',', nrows=n)
label_data = pd.read_csv(file_path5, sep=',', nrows=n)
miRNA_data = pd.read_csv(file_path6, sep=',', nrows=n)
mRNA_data = pd.read_csv(file_path7, sep=',', nrows=n)

A_data = pd.read_csv(file_path8, sep='\t', nrows=n)
labels_data = pd.read_csv(file_path9, sep='\t', nrows=n)
B_anno_data = pd.read_csv(file_path10, sep=',', nrows=n)

print(adjacency_data.shape)