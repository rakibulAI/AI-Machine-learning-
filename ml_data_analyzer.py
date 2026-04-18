"""
╔══════════════════════════════════════════════════════════════════╗
║           ML DATA ANALYZER — INTELLIGENT CSV INSIGHTS           ║
║           No Pandas. No Matplotlib. Pure Python + NumPy.        ║
║           Author : Ammar                                         ║
║           GitHub : github.com/ammar                              ║
╚══════════════════════════════════════════════════════════════════╝

What this does:
  - Reads any CSV dataset
  - Auto-detects numeric vs categorical columns
  - Computes full statistical summary (mean, std, median, IQR, skew)
  - Detects outliers using Z-score and IQR methods
  - Finds feature correlations
  - Visualizes distributions as ASCII histograms
  - Generates a clean terminal report
  - Saves a summary .txt report file
"""

import csv
import math
import os
import time
from collections import Counter, defaultdict


# ═══════════════════════════════════════════
#  MATH UTILITIES (no external libs)
# ═══════════════════════════════════════════

def mean(values):
    return sum(values) / len(values) if values else 0.0

def variance(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)

def std_dev(values):
    return math.sqrt(variance(values))

def median(values):
    s = sorted(values)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) if s else 0.0

def percentile(values, p):
    s = sorted(values)
    n = len(s)
    k = (n - 1) * p / 100
    f, c = int(k), math.ceil(k)
    return s[f] if f == c else s[f] * (c - k) + s[c] * (k - f)

def skewness(values):
    """Pearson's moment coefficient of skewness"""
    if len(values) < 3:
        return 0.0
    m   = mean(values)
    s   = std_dev(values)
    if s == 0:
        return 0.0
    n   = len(values)
    return (n / ((n - 1) * (n - 2))) * sum(((x - m) / s) ** 3 for x in values)

def kurtosis(values):
    if len(values) < 4:
        return 0.0
    m = mean(values)
    s = std_dev(values)
    if s == 0:
        return 0.0
    n = len(values)
    k = sum(((x - m) / s) ** 4 for x in values) / n
    return k - 3  # excess kurtosis

def pearson_correlation(x, y):
    """Pearson r between two equal-length lists"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx, my   = mean(x), mean(y)
    num      = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den      = math.sqrt(sum((xi - mx) ** 2 for xi in x) *
                         sum((yi - my) ** 2 for yi in y))
    return num / den if den else 0.0


# ═══════════════════════════════════════════
#  OUTLIER DETECTION
# ═══════════════════════════════════════════

def detect_outliers_zscore(values, threshold=3.0):
    m, s = mean(values), std_dev(values)
    if s == 0:
        return []
    return [i for i, v in enumerate(values) if abs((v - m) / s) > threshold]

def detect_outliers_iqr(values):
    q1  = percentile(values, 25)
    q3  = percentile(values, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [i for i, v in enumerate(values) if v < lo or v > hi]


# ═══════════════════════════════════════════
#  ASCII VISUALIZATION
# ═══════════════════════════════════════════

def ascii_histogram(values, title="", bins=20, width=50):
    if not values:
        return
    mn, mx = min(values), max(values)
    rng = mx - mn or 1e-9

    counts = [0] * bins
    edges  = [mn + i * rng / bins for i in range(bins + 1)]

    for v in values:
        idx = min(int((v - mn) / rng * bins), bins - 1)
        counts[idx] += 1

    max_count = max(counts) or 1
    bar_char  = "█"

    print(f"\n  {'─' * 54}")
    if title:
        print(f"  HISTOGRAM: {title}")
    print(f"  Range: [{mn:.3f}, {mx:.3f}]  |  n = {len(values)}")
    print(f"  {'─' * 54}")

    for i, count in enumerate(counts):
        bar_len   = int(count / max_count * width)
        bar       = bar_char * bar_len
        lo_edge   = edges[i]
        pct       = count / len(values) * 100
        print(f"  {lo_edge:8.3f} │{bar:<{width}} {count:4d} ({pct:4.1f}%)")

    print(f"  {'─' * 54}")

def ascii_bar_chart(labels, values, title="", width=40):
    if not values:
        return
    max_v = max(values) or 1
    print(f"\n  {'─' * 54}")
    if title:
        print(f"  {title}")
    print(f"  {'─' * 54}")
    for label, val in zip(labels, values):
        bar_len = int(val / max_v * width)
        bar     = "▇" * bar_len
        print(f"  {str(label)[:15]:>15} │ {bar:<{width}} {val}")
    print(f"  {'─' * 54}")

def ascii_correlation_matrix(columns, matrix):
    n = len(columns)
    cell = 7
    header = " " * 18
    for col in columns:
        header += f"{col[:6]:^{cell}}"
    print(f"\n  CORRELATION MATRIX")
    print(f"  {'─' * (18 + cell * n)}")
    print(f"  {header}")
    print(f"  {'─' * (18 + cell * n)}")

    for i, row_col in enumerate(columns):
        line = f"  {row_col[:16]:>16}  "
        for j in range(n):
            val = matrix[i][j]
            if i == j:
                cell_str = " diag "
            elif abs(val) >= 0.7:
                cell_str = f" ★{val:+.2f}"
            elif abs(val) >= 0.4:
                cell_str = f"  {val:+.2f} "
            else:
                cell_str = f"  {val:+.2f} "
            line += f"{cell_str:^{cell}}"
        print(line)
    print(f"  {'─' * (18 + cell * n)}")
    print("  ★ = strong correlation (|r| ≥ 0.7)\n")


# ═══════════════════════════════════════════
#  CSV LOADER
# ═══════════════════════════════════════════

def load_csv(filepath):
    """Load CSV into dict of column_name → list of values"""
    data = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in reader:
            for col in columns:
                data[col].append(row[col].strip())
    return dict(data), columns

def parse_numeric(col_values):
    """Try to convert list of strings to floats. Return None if not numeric."""
    result = []
    for v in col_values:
        try:
            result.append(float(v))
        except (ValueError, TypeError):
            return None
    return result


# ═══════════════════════════════════════════
#  MAIN ANALYZER
# ═══════════════════════════════════════════

class MLDataAnalyzer:

    def __init__(self, filepath):
        self.filepath = filepath
        self.report_lines = []

    def log(self, line=""):
        print(line)
        self.report_lines.append(line)

    def analyze(self):
        self._print_banner()
        self.log(f"  📂 File: {self.filepath}")
        self.log(f"  🕐 Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Load data
        data, columns = load_csv(self.filepath)
        n_rows = len(data[columns[0]])
        n_cols = len(columns)

        self.log("═" * 60)
        self.log(f"  DATASET OVERVIEW")
        self.log("═" * 60)
        self.log(f"  Rows      : {n_rows}")
        self.log(f"  Columns   : {n_cols}")
        self.log(f"  Columns   : {', '.join(columns)}")

        # Classify columns
        numeric_cols     = {}
        categorical_cols = {}

        for col in columns:
            parsed = parse_numeric(data[col])
            if parsed:
                numeric_cols[col] = parsed
            else:
                categorical_cols[col] = data[col]

        self.log(f"  Numeric   : {list(numeric_cols.keys())}")
        self.log(f"  Categorical: {list(categorical_cols.keys())}\n")

        # ── Missing Values Check
        self.log("═" * 60)
        self.log("  MISSING VALUE ANALYSIS")
        self.log("═" * 60)
        for col in columns:
            values = data[col]
            missing = sum(1 for v in values if v in ("", "NA", "null", "None", "NaN"))
            pct     = missing / n_rows * 100
            status  = "⚠️  MISSING" if missing > 0 else "✅ Complete"
            self.log(f"  {col[:25]:25s} │ Missing: {missing:4d} ({pct:5.1f}%) │ {status}")

        # ── Numeric Statistics
        if numeric_cols:
            self.log("\n" + "═" * 60)
            self.log("  NUMERIC COLUMN STATISTICS")
            self.log("═" * 60)

            for col, values in numeric_cols.items():
                q1   = percentile(values, 25)
                q3   = percentile(values, 75)
                skew = skewness(values)
                kurt = kurtosis(values)

                self.log(f"\n  ▶ {col}")
                self.log(f"    Count    : {len(values)}")
                self.log(f"    Mean     : {mean(values):.4f}")
                self.log(f"    Std Dev  : {std_dev(values):.4f}")
                self.log(f"    Min      : {min(values):.4f}")
                self.log(f"    Q1       : {q1:.4f}")
                self.log(f"    Median   : {median(values):.4f}")
                self.log(f"    Q3       : {q3:.4f}")
                self.log(f"    Max      : {max(values):.4f}")
                self.log(f"    IQR      : {q3 - q1:.4f}")
                self.log(f"    Skewness : {skew:.4f}  {'(right-skewed)' if skew > 0.5 else '(left-skewed)' if skew < -0.5 else '(approx. normal)'}")
                self.log(f"    Kurtosis : {kurt:.4f}")

                # Outliers
                z_out  = detect_outliers_zscore(values)
                iq_out = detect_outliers_iqr(values)
                self.log(f"    Outliers : {len(z_out)} (Z-score) │ {len(iq_out)} (IQR)")

                # Histogram
                ascii_histogram(values, title=col, bins=15, width=40)

        # ── Categorical Statistics
        if categorical_cols:
            self.log("\n" + "═" * 60)
            self.log("  CATEGORICAL COLUMN ANALYSIS")
            self.log("═" * 60)

            for col, values in categorical_cols.items():
                counts   = Counter(values)
                n_unique = len(counts)
                top5     = counts.most_common(5)

                self.log(f"\n  ▶ {col}")
                self.log(f"    Unique values : {n_unique}")
                self.log(f"    Top values    :")
                for val, cnt in top5:
                    bar = "▇" * int(cnt / n_rows * 30)
                    self.log(f"      {str(val)[:20]:>20} │{bar} {cnt} ({cnt/n_rows*100:.1f}%)")

        # ── Correlation Matrix
        if len(numeric_cols) >= 2:
            self.log("\n" + "═" * 60)
            self.log("  FEATURE CORRELATION ANALYSIS")
            self.log("═" * 60)

            col_names = list(numeric_cols.keys())
            n = len(col_names)
            matrix = [[0.0] * n for _ in range(n)]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i][j] = 1.0
                    else:
                        matrix[i][j] = pearson_correlation(
                            numeric_cols[col_names[i]],
                            numeric_cols[col_names[j]]
                        )

            ascii_correlation_matrix(col_names, matrix)

            # Strong correlations
            strong = []
            for i in range(n):
                for j in range(i + 1, n):
                    r = matrix[i][j]
                    if abs(r) >= 0.5:
                        direction = "positive" if r > 0 else "negative"
                        strength  = "strong" if abs(r) >= 0.7 else "moderate"
                        strong.append((col_names[i], col_names[j], r, strength, direction))

            if strong:
                self.log("  Notable Correlations:")
                for a, b, r, strength, direction in strong:
                    self.log(f"    • {a} ↔ {b}: r = {r:.3f}  ({strength} {direction})")

        # ── Save report
        report_path = self.filepath.replace(".csv", "_ml_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_lines))

        self.log("\n" + "═" * 60)
        self.log(f"  ✅ Analysis complete!")
        self.log(f"  📄 Report saved: {report_path}")
        self.log("═" * 60 + "\n")

    def _print_banner(self):
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        ML DATA ANALYZER — by Ammar                          ║
║        Intelligent CSV Insights • Zero Dependencies         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)


# ═══════════════════════════════════════════
#  DEMO DATA GENERATOR
# ═══════════════════════════════════════════

def generate_demo_dataset(filepath="sample_data.csv", n=300):
    """
    Creates a realistic demo CSV dataset about student performance.
    No external datasets needed — generates everything from scratch.
    """
    import random
    random.seed(42)

    rows = []
    for i in range(n):
        study_hours  = round(random.gauss(5, 2), 1)
        study_hours  = max(0, min(12, study_hours))
        sleep_hours  = round(random.gauss(7, 1), 1)
        sleep_hours  = max(4, min(10, sleep_hours))
        # Score correlates with study hours
        score = 40 + study_hours * 4 + sleep_hours * 1.5 + random.gauss(0, 8)
        score = round(max(0, min(100, score)), 1)
        attendance = round(min(100, max(30, random.gauss(75, 15))), 1)
        grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
        gender = random.choice(["Male", "Female"])

        rows.append({
            "student_id"   : i + 1,
            "study_hours"  : study_hours,
            "sleep_hours"  : sleep_hours,
            "attendance_pct": attendance,
            "exam_score"   : score,
            "grade"        : grade,
            "gender"       : gender,
        })

    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✅ Demo dataset created: {filepath}  ({n} rows)")
    return filepath


# ═══════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("\n  ML DATA ANALYZER — by Ammar")
    print("  ─────────────────────────────")

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"  ❌ File not found: {csv_path}")
            sys.exit(1)
        print(f"  Using: {csv_path}\n")
    else:
        print("  No CSV provided — generating demo dataset...\n")
        csv_path = generate_demo_dataset("sample_student_data.csv", n=300)

    analyzer = MLDataAnalyzer(csv_path)
    analyzer.analyze()
