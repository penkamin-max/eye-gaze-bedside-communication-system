import pandas as pd
import glob
import os


DATA_DIR = "gaze_dataset"

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if not files:
    print("Khong tim thay file CSV nao trong gaze_dataset.")
    print("Hay chay collect_gaze_data.py truoc.")
    exit()

df_list = []

for file in files:
    temp = pd.read_csv(file)
    temp["source_file"] = os.path.basename(file)
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

# Chi lay frame co mat, KHONG loc head_pose_status de tranh mat LEFT/RIGHT
df = df[df["valid_face"] == 1]

df = df.dropna(subset=["smooth_raw_x", "smooth_raw_y"])

if len(df) == 0:
    print("Khong co data hop le de phan tich.")
    exit()

summary = df.groupby("target_name").agg(
    mean_x=("smooth_raw_x", "mean"),
    mean_y=("smooth_raw_y", "mean"),
    std_x=("smooth_raw_x", "std"),
    std_y=("smooth_raw_y", "std"),
    min_x=("smooth_raw_x", "min"),
    max_x=("smooth_raw_x", "max"),
    min_y=("smooth_raw_y", "min"),
    max_y=("smooth_raw_y", "max"),
    count=("smooth_raw_x", "count")
).reset_index()

print("\n===== GAZE SUMMARY =====")
print(summary)

output_path = os.path.join(DATA_DIR, "gaze_summary.csv")
summary.to_csv(output_path, index=False)

print(f"\nDa luu ket qua trung binh vao: {output_path}")


# =========================
# AUTO GENERATE PYTHON BLOCK
# =========================

profile_lines = []
profile_lines.append("GAZE_PROFILES = {")

for _, row in summary.iterrows():
    target = row["target_name"]

    line = (
        f'    "{target}": '
        f'{{"x": {row["mean_x"]:.6f}, '
        f'"y": {row["mean_y"]:.6f}, '
        f'"std_x": {row["std_x"]:.6f}, '
        f'"std_y": {row["std_y"]:.6f}}},'
    )

    profile_lines.append(line)

profile_lines.append("}")

profile_text = "\n".join(profile_lines)

profile_path = os.path.join(DATA_DIR, "gaze_profiles_code.txt")

with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)

print(f"Da tao code profile tai: {profile_path}")
print("\n===== COPY BLOCK NAY VAO gaze_control.py =====")
print(profile_text)