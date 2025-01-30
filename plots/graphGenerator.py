import pandas as pd
import matplotlib.pyplot as plt
import os

# Percorso del file CSV (modifica questo percorso in base al tuo sistema)
#file_path = "./logs/Mnist/FP/0,25SF.csv"
file_path = "./logs/Mnist/EarlyDetection/0,75MF.csv"

# Carica il file CSV
data = pd.read_csv(file_path)

# Estrai la percentuale di attackers e i valori (in base alla cartella)
attackers_percentage = data["% of attackers"]
values = data.iloc[:, 1:].copy()  # Make a copy to modify

# Replace '*' with 31 for calculations, keep the '*' in the values for later annotation
values = values.applymap(lambda x: 31 if x == '*' else x)

# Convert to integers, keeping 31 for '*' cells
values = values.applymap(lambda x: int(x) if isinstance(x, (int, float)) else x)

# Determina il tipo di etichetta dell'asse Y e il titolo
folder_name = os.path.basename(os.path.dirname(file_path))
ylabel = "Early Detection Iteration" if folder_name == "EarlyDetection" else "False Positive Rate"
title = "Early Detection" if folder_name == "EarlyDetection" else "False Positive Rate"

# Estrai informazioni dal nome del file
file_name = os.path.basename(file_path)
flipping_type = "MultiFlipping" if "MF" in file_name else "SingleFlipping"
value_in_file = file_name.split(",")[1]

# Rimuovi il suffisso dal nome del file
for suffix in ["MF.csv", "SF.csv"]:
    if value_in_file.endswith(suffix):
        value_in_file = value_in_file.removesuffix(suffix)
        break

title += f" {float(value_in_file)}% {flipping_type}"

# Crea il grafico
plt.figure(figsize=(12, 6))

# Color map per ogni barra
num_columns = len(values.columns)
colors = plt.cm.get_cmap("tab10", num_columns)

# Larghezza delle barre
bar_width = 1
x_positions = attackers_percentage

# Disegna le barre con un colore diverso per ogni colonna
for idx, col in enumerate(values.columns):
    x_offset = (idx - (num_columns - 1) / 2) * (bar_width + 0.1)
    bar_values = values[col].astype(float)

    # Replace zero values with a small number to make them visible (e.g., 0.1)
    bar_values = bar_values.apply(lambda x: 0.5 if x == 0 else x)

    # Disegnare le barre
    bars = plt.bar(x_positions + x_offset, bar_values, width=bar_width, label=col, color=colors(idx))

    # Aggiungi asterischi sopra le barre che originariamente erano '*'
    for i, orig_val in enumerate(data[col]):
        if orig_val == "*":
            plt.text(x_positions.iloc[i] + x_offset, 31, "*", ha='center', va='bottom', fontsize=14, fontweight='bold')

# Aggiungere etichette
plt.xlabel("% attackers")
plt.ylabel(ylabel)

# Aggiungi il titolo
plt.title(title)

# Aggiungi la legenda
plt.legend(loc="upper right")

metric = "ED" if folder_name == "EarlyDetection" else "FP"
plot_name = metric + "_" + file_name.removesuffix('.csv')

if metric == "FP": 
    plt.ylim(top=100)

# Salva il grafico
plt.savefig(f"./plots/{plot_name}.png", dpi=300, bbox_inches="tight")

# Ottimizza la disposizione e mostra il grafico
plt.tight_layout()
plt.show()
