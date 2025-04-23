import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Load the CSV file
data = pd.read_csv('../data/Sri Lankan Dataset/Imagewise_Data.csv')  # Replace with actual file path

# Create a PDF file to save all plots
pdf_filename = "statistics_sri_lankan_imagewise.pdf"

# Category,Clinical Diagnosis,Lesion Annotation Count
with PdfPages(pdf_filename) as pdf:
    # Categorical columns for pie charts
    categorical_columns = ['Category', 'Clinical Diagnosis']
    for column in categorical_columns:
        plt.figure(figsize=(30, 30))
        value_counts = data[column].value_counts()
        labels = [f'{index} ({value})' for index, value in zip(value_counts.index, value_counts.values)]
        value_counts.plot.pie(autopct=lambda p: f'{p:.1f}%\n({int(p*sum(value_counts)/100)})', startangle=0, cmap='Pastel1', labels=labels)
        plt.title(f'Distribution of {column}')
        plt.ylabel('')  # Hide y-label for better visibility
        pdf.savefig()  # Save the figure to the PDF
        plt.close()
    
    # Numerical columns for bar plots and statistics
    numerical_columns = ['Lesion Annotation Count']
    for column in numerical_columns:
        plt.figure(figsize=(8, 5))
        ax = sns.histplot(data[column], bins=10, kde=True, color='skyblue')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10, color='black')
        plt.axvline(data[column].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {data[column].mean():.2f}')
        plt.axvline(data[column].median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {data[column].median():.2f}')
        plt.axvline(data[column].mode()[0], color='blue', linestyle='dashed', linewidth=2, label=f'Mode: {data[column].mode()[0]:.2f}')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        pdf.savefig()  # Save the figure to the PDF
        plt.close()
    
    # Save statistical details as a text page in PDF
    with open("stats.txt", "w") as f:
        f.write(f"Total number of entries: {len(data)}\n\n")
        for column in numerical_columns:
            f.write(f"Statistics for {column}:\n")
            f.write(f"Mean: {data[column].mean():.2f}\n")
            f.write(f"Median: {data[column].median():.2f}\n")
            f.write(f"Mode: {data[column].mode()[0]:.2f}\n")
            f.write(f"Standard Deviation: {data[column].std():.2f}\n")
            f.write(f"Max: {data[column].max():.2f}\n")
            f.write(f"Min: {data[column].min():.2f}\n\n")
    
    # Add the statistics text page to the PDF
    plt.figure(figsize=(8, 8))
    plt.text(0.1, 0.5, open("stats.txt").read(), fontsize=12, family='monospace')
    plt.axis('off')
    pdf.savefig()
    plt.close()

print(f"All plots saved in {pdf_filename}")