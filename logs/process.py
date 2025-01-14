import re
import json
import glob
import os

def parse_filename(filename):
    pattern = r"^(\d+)frames\.txt_(\d+)\.txt$"
    basename = os.path.basename(filename)
    match = re.match(pattern, basename)
    if match:
        frame = int(match.group(1))
        seq_len = int(match.group(2))
        return frame, seq_len
    else:
        return None, None

def extract_last_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    # 修改正则表达式以匹配完整的含有四个组件的JSON
    json_blocks = re.findall(r'\{[\s\S]*?"preprocess"[\s\S]*?"MLP"[\s\S]*?"parallel_attention"[\s\S]*?"VAE"[\s\S]*?\}', content)
    if not json_blocks:
        return None
    last_block = json_blocks[-1]
    last_block += "}"
    return json.loads(last_block)


def main():
    # 获取所有GPU型号目录
    gpu_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    
    print("gpu,frame,seq_len,attn_time,mlp_time,attn_percent,mlp_percent,vae_time,preprocess_time")

    for gpu_dir in gpu_dirs:
        # 遍历每个GPU目录下的txt文件
        txt_files = glob.glob(os.path.join(gpu_dir, "*frames.txt_*.txt"))
        
        for txt_file in reversed(txt_files):
            frame, seq_len = parse_filename(txt_file)
            if frame is None:
                continue

            data_dict = extract_last_json(txt_file)
            if not data_dict:
                continue

            attn_time = data_dict["parallel_attention"]["total_time"]
            mlp_time = data_dict["MLP"]["total_time"]
            vae_time = data_dict["VAE"]["total_time"]
            preprocess_time = data_dict["preprocess"]["total_time"]

            sum_time = attn_time + mlp_time
            if sum_time > 0:
                attn_percent = attn_time / sum_time
                mlp_percent = mlp_time / sum_time
            else:
                attn_percent = 0
                mlp_percent = 0

            print(f"{gpu_dir},{frame},{seq_len},"
                  f"{attn_time:.4f},{mlp_time:.4f},"
                  f"{attn_percent:.4f},{mlp_percent:.4f},"
                  f"{vae_time:.4f},{preprocess_time:.4f}")

if __name__ == "__main__":
    main()