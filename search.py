import os
import pandas as pd
import random


def generate_test_database(filename="large_database.txt", target_size_gb=0.1):
  """Generates a test database with controlled ranges to guarantee matches."""
  target_bytes = target_size_gb * 1024 * 1024 * 1024
  print(
      f"Generating {target_size_gb} GB dataset: {filename} (this may take a"
      " moment)..."
  )

  header = "# x y v A_real A_imag\n"
  current_size = len(header.encode("utf-8"))

  with open(filename, "w") as f:
    f.write(header)
    while current_size < target_bytes:
      chunk_lines = []
      for _ in range(10000):  # Batch of 10,000 rows
        x = round(random.uniform(-10, 10), 4)
        y = round(random.uniform(-10, 10), 4)

        # Force a precise match occasionally for testing purposes
        if random.random() < 0.001:
          v = 12.5  # Guaranteed match value
          a_real, a_imag = 0.9, 0.9  # Guaranteed |A|^2 > 0.8
        else:
          v = round(x**2 + y**2 + random.uniform(-0.05, 0.05), 4)
          a_real = round(random.uniform(-1, 1), 4)
          a_imag = round(random.uniform(-1, 1), 4)

        line = f"{x:10.4f} {y:10.4f} {v:10.4f} {a_real:10.4f} {a_imag:10.4f}\n"
        chunk_lines.append(line)

      chunk_str = "".join(chunk_lines)
      f.write(chunk_str)
      current_size += len(chunk_str.encode("utf-8"))

  print(
      f"Dataset generated successfully! Final size:"
      f" {os.path.getsize(filename) / (1024**3):.2f} GB"
  )


def search_large_text_file(
    filepath, v_target=12.5, tau=0.5, tolerance=1e-2, chunk_size=100000
):
  """Scans large text files chunk-by-chunk under strict memory constraints to solve:

  (x*, y*) = arg find [(v(x,y) == V_target) & (|A(x,y)|^2 > tau)]
  """
  print(f"\nScanning '{filepath}' for v_target={v_target}, tau={tau}...")

  matches = []
  # Replaced delim_whitespace=True with sep='\s+' for compatibility with newer Pandas versions
  chunk_iter = pd.read_csv(
      filepath,
      comment="#",
      sep=r"\s+",
      header=None,
      names=["x", "y", "v", "A_real", "A_imag"],
      chunksize=chunk_size,
  )

  total_rows_scanned = 0

  for chunk in chunk_iter:
    total_rows_scanned += len(chunk)

    # 1. Compute squared magnitude: |A|^2 = real^2 + imag^2
    intensity = chunk["A_real"] ** 2 + chunk["A_imag"] ** 2

    # 2. Evaluate conditions
    cond_v = (chunk["v"] - v_target).abs() <= tolerance
    cond_a = intensity > tau

    # 3. Filter rows matching both criteria
    valid_rows = chunk[cond_v & cond_a]

    if not valid_rows.empty:
      for _, row in valid_rows.iterrows():
        matches.append(
            {
                "x": row["x"],
                "y": row["y"],
                "v": row["v"],
                "intensity": intensity[row.name],
            }
        )

  print(f"Scanned {total_rows_scanned:,} total rows.")

  if not matches:
    print("No matching states found in the database.")
    return None

  print(f"Found {len(matches)} matching record(s)!")
  best_match = matches[0]
  print(
      f"Optimal Solution Selected: (x*, y*) = ({best_match['x']}, {best_match['y']})"
      f" | v = {best_match['v']} | |A|^2 = {best_match['intensity']:.4f}"
  )

  return (best_match["x"], best_match["y"])


# --- Example Execution ---
if __name__ == "__main__":
  filename = "large_database.txt"

  # 1. Generate the test file
  generate_test_database(filename, target_size_gb=0.5)

  # 2. Run the memory-efficient chunked search algorithm
  optimal_coords = search_large_text_file(filename, v_target=12.5, tau=0.8)
  print(f"\nOptimal Solution Selected: {optimal_coords}")
