import pandas as pd
import requests
import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# 1. 填入你的 CSV 绝对路径 (注意路径前的 r)
input_file = r"data\cdk2.csv"
df_input = pd.read_csv(input_file)

if "chembl_id" not in df_input.columns:
    raise ValueError(f"文件中未找到 'chembl_id' 列。")

unique_ids = df_input["chembl_id"].dropna().unique().tolist()
print(f"共发现 {len(unique_ids)} 个唯一的 ChEMBL ID，开始拉取...")

id_to_smiles = {}
# 关键修复 1：严格遵守 ChEMBL API 的硬性限制，每批最多 20 个
batch_size = 20  
base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

for i in tqdm(range(0, len(unique_ids), batch_size), desc="Fetching SMILES"):
    batch_ids = unique_ids[i : i + batch_size]
    params = {
        "molecule_chembl_id__in": ",".join(batch_ids),
        "only": "molecule_chembl_id,molecule_structures"
    }
    
    # 关键修复 2：加入服务器 500 报错的自动重试机制 (最多重试 5 次)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                molecules = data.get("molecules", [])
                for mol_data in molecules:
                    cid = mol_data.get("molecule_chembl_id")
                    structs = mol_data.get("molecule_structures")
                    if structs and structs.get("canonical_smiles"):
                        id_to_smiles[cid] = structs["canonical_smiles"]
                break  # 成功拉取，跳出重试循环，进入下一个批次
            else:
                if attempt == max_retries - 1:
                    print(f"\n[Warning] 批次 {i} 重试 {max_retries} 次后依然失败，状态码: {response.status_code}")
                else:
                    time.sleep(2)  # 等待 2 秒后重试
                    
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[Error] 批次 {i} 网络请求崩溃: {e}")
            else:
                time.sleep(2)
                
    time.sleep(0.1) # 批次之间的礼貌性停顿

df_input["smiles"] = df_input["chembl_id"].map(id_to_smiles)
missing_count = df_input["smiles"].isna().sum()
print(f"\n拉取完成！完全缺失 SMILES 的分子数量: {missing_count} / {len(df_input)}")

# 3. 计算连续型物理化学特征
print("开始计算物理化学特征...")
features_list = []

for idx, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Computing Features"):
    smiles = row["smiles"]
    chembl_id = row["chembl_id"]
    label = row.get("label", None) 
    
    if pd.isna(smiles):
        continue
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
        
    features = {
        "chembl_id": chembl_id,
        "smiles": smiles,
        "MW (g/mol)": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "Rotatable_Bonds": Descriptors.NumRotatableBonds(mol),
        "Aromatic_Rings": Descriptors.NumAromaticRings(mol),
        "Heavy_Atoms": Descriptors.HeavyAtomCount(mol),
        "QED": QED.qed(mol)
    }
    if label is not None:
        features["label"] = label
    features_list.append(features)

df_features = pd.DataFrame(features_list)
if "label" in df_features.columns:
    cols = [c for c in df_features.columns if c != "label"] + ["label"]
    df_features = df_features[cols]

output_file = r"cdk2_features.csv"
df_features.to_csv(output_file, index=False)

print(f"\n大功告成！成功处理并保存 {len(df_features)} 个分子至 {output_file}")