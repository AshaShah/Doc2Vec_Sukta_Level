# Read all 1028 lines into a list
with open("Griff_translation.txt", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

# Ensure you actually have 1028 lines
if len(lines) != 1028:
    raise ValueError(f"Expected 1028 lines, found {len(lines)}")

# Groups based on line numbers (1-indexed)
groups = [
    (1, 191),
    (192, 234),
    (235, 296),
    (297, 354),
    (355, 441),
    (442, 516),
    (517, 620),
    (621, 723),
    (724, 837),
    (838, 1028)
]

output_lines = []

for start, end in groups:
    # Convert to 0-index: start-1 : end
    merged = " ".join(lines[start-1 : end])
    output_lines.append(merged)

# Write final grouped file
with open("mandala_Griffith.txt", "w", encoding="utf-8") as out:
    for line in output_lines:
        out.write(line + "\n")

print("Created sukta_grouped.txt with", len(output_lines), "lines.")








# # processed/sukta_*.txt → sukta_grouped.txt (one line per group of suktas)
# import re
# import glob

# # Helper to load one sukta and collapse it to a single line
# def load_one_line(fname):
#     with open(fname, "r", encoding="utf-8") as f:
#         text = f.read().strip()
#         return " ".join(text.split())   # collapse multiline → single-line

# # Groups you provided
# groups = [
#     (1, 191),
#     (192, 234),
#     (235, 296),
#     (297, 354),
#     (355, 441),
#     (442, 516),
#     (517, 620),
#     (621, 723),
#     (724, 837),
#     (838, 1028)
# ]

# output_lines = []

# for start, end in groups:
#     merged = []
#     for i in range(start, end + 1):
#         fname = f"processed/sukta_{i:04d}.txt"
#         one = load_one_line(fname)
#         merged.append(one)
#     # join all suktas of this group into one single line
#     line = " ".join(merged)
#     output_lines.append(line)

# # Write final file
# with open("mandala.txt", "w", encoding="utf-8") as out:
#     for line in output_lines:
#         out.write(line + "\n")

# print("Created mandala.txt with", len(output_lines), "lines.")







# processed/sukta_*.txt → sukta.txt (one line per sukta)

# import glob
# import re

# # Collect all sukta files
# files = sorted(
#     glob.glob("processed/sukta_*.txt"),
#     key=lambda f: int(re.search(r"(\d+)", f).group(1))
# )

# with open("sukta.txt", "w", encoding="utf-8") as out:
#     for f in files:
#         with open(f, "r", encoding="utf-8") as inp:
#             text = inp.read().strip()
#             # collapse multiple lines or spaces into one line
#             one_line = " ".join(text.split())
#             out.write(one_line + "\n")

# print("DONE: Created sukta.txt with", len(files), "lines.")
