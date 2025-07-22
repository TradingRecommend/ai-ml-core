import pandas as pd
import json
import csv

# Đọc CSV
df = pd.read_csv('/Users/Programming/DS/stock-recommend/calculate/data.csv')  # Phải có cột 'apartment', 'acreage'
acreage_dict = df.set_index('apartment')['acreage'].to_dict()

# Đọc file JSON đầu vào
with open('/Users/Programming/DS/stock-recommend/calculate/input.json', 'r', encoding='utf-8') as f:
    all_json_data = json.load(f)

# Kết quả tổng hợp
final_result = {}

# Lặp qua từng nhóm ("1", "2", ..., "26")
for group_id, group_data in all_json_data.items():
    group_result = {}
    for key, apt_list in group_data.items():
        total = sum(acreage_dict.get(apt, 0) for apt in apt_list)
        group_result[key] = total
    final_result[group_id] = group_result

# Ghi kết quả ra file
with open('/Users/Programming/DS/stock-recommend/calculate/result.json', 'w') as f:
    json.dump(final_result, f, indent=2)

print("✅ Đã ghi kết quả vào 'result.json'")


# Bước 1: Đọc data.csv thành dict {apartment: acreage}
acreage_map = {}
with open("/Users/Programming/DS/stock-recommend/calculate/data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        apt = int(row["apartment"])
        acreage_map[apt] = float(row["acreage"])

expected_values = set(range(1, 17))  # Tập các giá trị cần có
group_missing_acreage = {}

# Bước 3: Kiểm tra thiếu và tính tổng
for group_id, subgroup in all_json_data.items():
    appeared = set()
    for _, apartments in subgroup.items():
        appeared.update(apartments)

    missing = expected_values - appeared
    missing_acreage = sum(acreage_map.get(apt, 0) for apt in missing)

    if missing:
        group_missing_acreage[group_id] = {
            "missing": sorted(missing),
            "total_missing_acreage": round(missing_acreage, 2)
        }

# Bước 4: Ghi ra file kết quả
with open("/Users/Programming/DS/stock-recommend/calculate/missing_acreage_result.json", "w", encoding="utf-8") as f:
    json.dump(group_missing_acreage, f, indent=2, ensure_ascii=False)

print("✅ Kết quả đã lưu vào 'missing_acreage_result.json'")


# Calculate total missing acreage
total_missing_acreage = sum(group["total_missing_acreage"] for group in group_missing_acreage.values())

# Print or save result
print("Total missing acreage across all groups:", total_missing_acreage)