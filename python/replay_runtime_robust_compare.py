#!/usr/bin/env python3
"""
Replay a processed CSV through generated C++ legacy and robust runtime modes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


HARNESS_SOURCE = r"""
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "sevendofDynamics.h"

namespace
{
constexpr EcSizeT kNumJoints = 7;

void readJointVector(std::istream& input, EcRealVector& values)
{
    values.assign(kNumJoints, 0.0);
    for (EcSizeT idx = 0; idx < kNumJoints; ++idx)
    {
        input >> values[idx];
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <gx> <gy> <gz> <output_joint>\n";
        return 2;
    }

    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
        std::cerr << "failed to open input file\n";
        return 3;
    }

    const EcReal gx = std::stod(argv[2]);
    const EcReal gy = std::stod(argv[3]);
    const EcReal gz = std::stod(argv[4]);
    const EcSizeT jointIndex = static_cast<EcSizeT>(std::stoul(argv[5]));

    sevendofDynamics dynamics;
    dynamics.setGravityVector(gx, gy, gz);

    RuntimeNoTimeConfig runtimeCfg;
    runtimeCfg.enableInputValidation = true;
    runtimeCfg.enableDdqClamp = true;
    runtimeCfg.enableFrictionStateMachine = true;
    runtimeCfg.enableHoldMode = true;
    runtimeCfg.enableMCGate = true;
    dynamics.setRuntimeNoTimeConfig(runtimeCfg);

    DdqClampConfig clampCfg;
    clampCfg.enabled = true;
    clampCfg.ddqLimit.assign(kNumJoints, 10.0);
    dynamics.setDdqClampConfig(clampCfg);

    FrictionStateConfig frictionCfg;
    frictionCfg.enabled = true;
    frictionCfg.enableHysteresis = true;
    frictionCfg.enableSmoothSign = false;
    frictionCfg.enableHoldMode = true;
    frictionCfg.vOff.assign(kNumJoints, 0.01);
    frictionCfg.vOn.assign(kNumJoints, 0.03);
    frictionCfg.aOff.assign(kNumJoints, 0.2);
    frictionCfg.smoothVelocity.assign(kNumJoints, 0.02);
    frictionCfg.holdBias.assign(kNumJoints, 0.0);
    dynamics.setFrictionStateConfig(frictionCfg);

    MCGateConfig gateCfg;
    gateCfg.enabled = true;
    gateCfg.staticWM = 0.0;
    gateCfg.staticWC = 0.0;
    gateCfg.transitionWM = 0.0;
    gateCfg.transitionWC = 0.0;
    gateCfg.dynamicWM = 1.0;
    gateCfg.dynamicWC = 1.0;
    dynamics.setMCGateConfig(gateCfg);

    EcRealVector q, dq, ddq, tauLegacy, tauRobust;
    RuntimeDiagnostics diagnostics;

    std::cout << std::setprecision(17);
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::istringstream row(line);
        readJointVector(row, q);
        readJointVector(row, dq);
        readJointVector(row, ddq);

        dynamics.setRuntimeTorqueMode(LEGACY_RAW);
        if (!dynamics.calculateEstimateJointToqrues(q, dq, ddq, EcRealVector(), tauLegacy))
        {
            return 4;
        }
        dynamics.setRuntimeTorqueMode(ROBUST_NO_TIME);
        if (!dynamics.calculateEstimateJointToqrues(q, dq, ddq, EcRealVector(), tauRobust))
        {
            return 5;
        }
        dynamics.getLastRuntimeDiagnostics(diagnostics);

        std::cout
            << tauLegacy[jointIndex] << " "
            << diagnostics.tau_M_raw[jointIndex] << " "
            << diagnostics.tau_C_raw[jointIndex] << " "
            << diagnostics.tau_G[jointIndex] << " "
            << diagnostics.tau_F_total[jointIndex] << " "
            << tauRobust[jointIndex] << " "
            << diagnostics.friction_sign[jointIndex] << "\n";
    }
    return 0;
}
"""


def _write_input(df: pd.DataFrame, path: Path, num_joints: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            values: list[str] = []
            for prefix in ("q", "dq", "ddq"):
                for joint_idx in range(1, num_joints + 1):
                    values.append(f"{float(row[f'{prefix}_{joint_idx}']):.17g}")
            handle.write(" ".join(values))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay processed CSV through generated C++ legacy and robust modes.")
    parser.add_argument("--processed-path", required=True)
    parser.add_argument("--generated-cpp", default=str(PROJECT_ROOT / "output" / "sevendofDynamics.cpp"))
    parser.add_argument("--joint-index", type=int, default=3, help="1-based joint index to report.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit.")
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "output" / "diagnostics" / "runtime_robust_compare.json"))
    parser.add_argument("--output-csv", default=str(PROJECT_ROOT / "output" / "diagnostics" / "runtime_robust_compare.csv"))
    args = parser.parse_args()

    processed_path = Path(args.processed_path)
    generated_cpp = Path(args.generated_cpp)
    joint_index = int(args.joint_index) - 1

    df = pd.read_csv(processed_path)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    if df.empty:
        raise ValueError("processed CSV is empty")

    gravity = np.array(
        [
            float(df.iloc[0].get("gravity_x", -9.81)),
            float(df.iloc[0].get("gravity_y", 0.0)),
            float(df.iloc[0].get("gravity_z", 0.0)),
        ],
        dtype=float,
    )
    tau_measured = df[f"tau_{joint_index + 1}"].to_numpy(dtype=float)

    diagnostics_dir = Path(args.output_json).resolve().parent
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="runtime_replay_", dir=str(PROJECT_ROOT / "output" / "test")) as tmpdir:
        tmpdir_path = Path(tmpdir)
        harness_cpp = tmpdir_path / "runtime_replay_harness.cpp"
        harness_bin = tmpdir_path / "runtime_replay_harness"
        input_txt = tmpdir_path / "samples.txt"
        harness_cpp.write_text(HARNESS_SOURCE, encoding="utf-8")
        _write_input(df, input_txt, 7)

        command = [
            "g++",
            "-std=c++17",
            "-O2",
            "-I",
            str(PROJECT_ROOT / "output" / "test" / "rtos_stub"),
            "-I",
            str(generated_cpp.parent),
            str(generated_cpp),
            str(harness_cpp),
            "-o",
            str(harness_bin),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        result = subprocess.run(
            [
                str(harness_bin),
                str(input_txt),
                f"{gravity[0]:.17g}",
                f"{gravity[1]:.17g}",
                f"{gravity[2]:.17g}",
                str(joint_index),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    rows = []
    for raw_line in result.stdout.strip().splitlines():
        parts = raw_line.split()
        if len(parts) != 7:
            continue
        rows.append([float(part) for part in parts])

    report_df = pd.DataFrame(
        rows,
        columns=[
            "tau_legacy",
            "tau_M_robust",
            "tau_C_robust",
            "tau_G_robust",
            "tau_F_robust",
            "tau_total_robust",
            "friction_sign",
        ],
    )
    report_df["tau_measured"] = tau_measured[: len(report_df)]
    report_df["legacy_step_jump"] = report_df["tau_legacy"].diff().abs().fillna(0.0)
    report_df["robust_step_jump"] = report_df["tau_total_robust"].diff().abs().fillna(0.0)
    report_df["legacy_abs_error"] = (report_df["tau_legacy"] - report_df["tau_measured"]).abs()
    report_df["robust_abs_error"] = (report_df["tau_total_robust"] - report_df["tau_measured"]).abs()

    summary = {
        "processed_path": str(processed_path),
        "generated_cpp": str(generated_cpp),
        "joint_index_1based": joint_index + 1,
        "num_rows": int(len(report_df)),
        "friction_sign_switches": int((report_df["friction_sign"].diff().abs() > 1e-12).sum()),
        "legacy_step_jump_count_gt_0p2": int((report_df["legacy_step_jump"] > 0.2).sum()),
        "robust_step_jump_count_gt_0p2": int((report_df["robust_step_jump"] > 0.2).sum()),
        "legacy_rmse": float(np.sqrt(np.mean((report_df["tau_legacy"] - report_df["tau_measured"]) ** 2))),
        "robust_rmse": float(np.sqrt(np.mean((report_df["tau_total_robust"] - report_df["tau_measured"]) ** 2))),
        "legacy_mae": float(np.mean(report_df["legacy_abs_error"])),
        "robust_mae": float(np.mean(report_df["robust_abs_error"])),
    }

    report_df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved replay traces to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
