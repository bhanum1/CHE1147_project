import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# 1. Define your file paths here
target_file = r"data\egfr_erbB1.csv"  # Change this if your file is named differently (e.g., "egfr_erbB1.csv")
chemreps_file = r"chembl_37_chemreps.txt.gz"
output_file = r"egfr_erbB1_features.csv"

# Load the target dataset
df_input = pd.read_csv(target_file)
if "chembl_id" not in df_input.columns:
    raise ValueError(f"Column 'chembl_id' not found in {target_file}.")
print(f"Loaded {len(df_input)} target molecules from {target_file}")

# 2. 载入 ChEMBL 本地字典
print(f"正在加载本地 ChEMBL 字典，请稍候...")
df_dict = pd.read_csv(
    chemreps_file, 
    sep='\t', 
    usecols=['chembl_id', 'canonical_smiles']
)

# 3. 本地关系型拼接 (Left Join)
print("正在匹配 SMILES 结构...")
df_merged = pd.merge(df_input, df_dict, on="chembl_id", how="left")
df_merged.rename(columns={'canonical_smiles': 'smiles'}, inplace=True)

# 4. 计算连续型物理化学特征
print("开始计算物理化学特征...")
features_list = []

for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Computing Features"):
    smiles = row["smiles"]
    chembl_id = row["chembl_id"]
    
    # 【关键修复】：兼容读取 'is_inhibitor' 或 'label' 列
    raw_label = row.get("is_inhibitor", row.get("label", None))
    
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
    
    # 强制将提取到的活性标签转换为整数，并统一命名为 is_inhibitor 存入
    if raw_label is not None and not pd.isna(raw_label):
        features["is_inhibitor"] = int(raw_label)
        
    features_list.append(features)

# 5. 格式化并导出 CSV
df_features = pd.DataFrame(features_list)

# 将 'is_inhibitor' 列强制移动到 DataFrame 的最后一列
if "is_inhibitor" in df_features.columns:
    cols = [c for c in df_features.columns if c != "is_inhibitor"] + ["is_inhibitor"]
    df_features = df_features[cols]

df_features.to_csv(output_file, index=False)
print(f"\n大功告成！成功处理并保存 {len(df_features)} 个分子至 {output_file}")