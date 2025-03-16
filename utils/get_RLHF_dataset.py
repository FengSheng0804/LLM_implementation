import pyarrow.parquet as pq
import pandas as pd

# 打开parquet文件
chosen = []
rejected = []

# 读取四个文件
for i in range(1, 5):
    df = pq.read_table("F:/desktop/" + str(i) + ".parquet").to_pandas()
    # 读取chosen和rejected列，将其转换为list
    chosen.extend(df['chosen'].values.tolist())
    rejected.extend(df['rejected'].values.tolist())

# 将chosen和rejected转换为DataFrame，然后保存为jsonl文件
df = pd.DataFrame({'chosen': chosen, 'rejected': rejected})
df.to_json("F:/desktop/myRLHF.jsonl", orient='records', lines=True)