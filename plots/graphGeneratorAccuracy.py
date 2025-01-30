import pandas as pd
import matplotlib.pyplot as plt
import os

# Percorso del file CSV (modifica questo percorso in base al tuo sistema)
file_path = "./logs/FMnist/Accuracy/0,50MF.csv"

# Carica il file CSV
data = pd.read_csv(file_path)

# Estrai la percentuale di attackers e i valori delle metriche
attackers_percentage = data["% of attackers"]
values = data.iloc[:, 1:]  # Assumiamo che le colonne dopo "% of attackers" siano le metriche

# Determina il tipo di etichetta dell'asse Y e il titolo
folder_name = os.path.basename(os.path.dirname(file_path))

if folder_name == "Accuracy":
    ylabel = "Accuracy"
    title = "Accuracy"
else:
    raise ValueError(f"Questo script è pensato per l'accuracy, ma il file è nella cartella: {folder_name}")

# Estrai informazioni dal nome del file
file_name = os.path.basename(file_path)
flipping_type = "MultiFlipping" if "MF" in file_name else "SingleFlipping"
value_in_file = file_name.split(",")[1]  # Ottieni il valore prima della virgola (ad esempio 0,25)

# Rimuovi il suffisso dal nome del file
for suffix in ["MF.csv", "SF.csv"]:
    if value_in_file.endswith(suffix):
        value_in_file = value_in_file.removesuffix(suffix)
        break

title += f" {float(value_in_file)}% {flipping_type}"

# Crea il grafico a linee
plt.figure(figsize=(10, 6))

# Colormap per differenziare le linee
num_columns = len(values.columns)
colors = plt.cm.get_cmap("tab10", num_columns)  # Usa una colormap per assegnare colori diversi

# Disegna le linee con marcatori per ogni metrica
for idx, col in enumerate(values.columns):
    plt.plot(attackers_percentage, values[col], 
             marker='o', linestyle='-', color=colors(idx), 
             label=col, linewidth=2, markersize=6)  # Linea più spessa e con punti visibili

# Aggiungi le etichette
plt.xlabel("% attackers")
plt.ylabel(ylabel + "( %)")
plt.ylim(bottom=0)
plt.title(title)

# Aggiungi la griglia per una migliore leggibilità
plt.grid(True, linestyle='--', alpha=0.6)

# Aggiungi la legenda
plt.legend( loc="best")

plot_name = "Acc" + "_" + file_name.removesuffix('.csv')
# Save the plot
plt.savefig(f"./plots/FMnist/{plot_name}.png", dpi=300, bbox_inches="tight")

# Mostra il grafico
plt.show()
