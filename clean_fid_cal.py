from cleanfid import fid

# 指定真实图像和生成图像的文件夹路径

def batch_process(real,fake):
    fid_score = fid.compute_fid(
        real,
        fake,
        mode="clean",  # 使用优化后的预处理
        num_workers=4,  # 并行处理加速
        model_name="clip_vit_b_32"
    )
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    import sys
    batch_process(sys.argv[1], sys.argv[2])