import pandas as pd
import os
from tabulate import tabulate
import glob
import matplotlib.pyplot as plt

# Percorso del file CSV (modifica questo percorso in base al tuo sistema)
folder_path = "./plots/Mnist/ASR/"
file = "0,50" # Cambialo in base ai file (0,25SF, 0,50SF, 0,75SF)

# 🔍 Trova tutti i file CSV nella cartella
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 🏗 Struttura per salvare i dati combinati
combined_data = {}

# 🔄 Processa ogni file CSV
for file_path in csv_files:
    if file in file_path:
        # 📜 Estrai il nome del file
        file_name = os.path.basename(file_path)

        # 📂 Estrai il nome del dataset (puoi modificarlo per altri dataset)
        dataset_name = "MNIST"  

        # 📊 Determina se è MF o SF
        flipping_type = "MF" if "MF" in file_name else "SF"

        # 🔢 Ottieni la percentuale di attackers dal nome del file
        try:
            attack_percentage = float(file_name.split(",")[0]) * 100
        except ValueError:
            print(f"⚠️ Errore nel parsing della percentuale da {file_name}")
            continue

        # 📖 Carica il file CSV
        data = pd.read_csv(file_path)

        # 📌 Aggiungi i dati alla struttura combinata
        for index, row in data.iterrows():
            key = (dataset_name, row["% of attackers"])  # Raggruppa per dataset e % attackers
            if key not in combined_data:
                combined_data[key] = {}

            # Aggiungi i valori per MF o SF
            for metric in data.columns[1:]:  # Esclude la colonna "% of attackers"
                column_name = f"{metric} {flipping_type}"
                combined_data[key][column_name] = row[metric]

# 📊 Converti i dati in DataFrame
df_combined = pd.DataFrame.from_dict(combined_data, orient="index").reset_index()
df_combined.columns = ["Dataset", "% Attacker"] + list(df_combined.columns[2:])  # Rinomina le colonne

# 📄 Salva la tabella in un file di testo
output_txt = f"./logs/{dataset_name}_{file}_ARStable.txt"
with open(output_txt, "w") as f:
    f.write(tabulate(df_combined, headers="keys", tablefmt="pretty", showindex=False))

# 🎨 Creazione dell'immagine della tabella
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("tight")
ax.axis("off")

# 🏷 Titolo dinamico
title = f"Attack Success Rate - MultiFlipping & SingleFlipping"

# 📊 Disegna la tabella con `matplotlib`
table = ax.table(cellText=df_combined.values, colLabels=df_combined.columns, cellLoc="center", loc="center")

# 🎨 Stile della tabella
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([i for i in range(len(df_combined.columns))])  # Imposta larghezza colonna

# 📄 Salva l'immagine della tabella
output_img = f"./logs/{dataset_name}_{file}_ARStable.png"
plt.savefig(output_img, bbox_inches="tight", dpi=300)
plt.close()

# 🖥 Messaggi di conferma
print(f"✅ Tabella testuale salvata in {output_txt}")
print(f"🖼️ Tabella come immagine salvata in {output_img}")

