# %%
import pandas as pd
import numpy as np
#%%
df = pd.read_csv('../../data/interim/cleaned_data.csv', sep=',')
df.head()

# %%
# Removendo o molecule_chembl_id, Unnamed:0, standard_value e faixa
df2 = df[['canonical_smiles','pic50']].copy()
df2.head()
# %%
features_lista =[]
smiles_validos =[]
pic50_validos =[]

#%%
from morgan_descriptors_generator import descritores

for idx, row in df2.iterrows():
    smiles = row['canonical_smiles']
    pic50 = row['pic50']

    features = descritores(smiles)

    if features is not None:
        features_lista.append(features)
        smiles_validos.append(smiles)
        pic50_validos.append(pic50)
    else:
        print(f" Smiles invalido na linha: {idx}: {smiles}")

X = np.array(features_lista)
y = np.array(pic50_validos)
              
# %%
print(f"\n{'='*60}")
print("RESULTADO:")
print(f"{'='*60}")
print(f"Moléculas processadas: {len(features_lista)}/{len(df2)}")
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"Features totais: {X.shape[1]} (2048 fp + 15 desc)")

print(f"\nVerificação de dados:")
print(f"  NaN em X: {np.isnan(X).sum()}")
print(f"  Inf em X: {np.isinf(X).sum()}")
print(f"  NaN em y: {np.isnan(y).sum()}")

# %%
'''
Salvando as variaveis processadas num parquet
'''
import pyarrow as pa
import pyarrow.parquet as pq

df_processado = pd.DataFrame(X)
df_processado['pic50'] = y
df_processado['smiles'] = smiles_validos

df_processado.to_parquet('../../data/processed.parquet', index=False)