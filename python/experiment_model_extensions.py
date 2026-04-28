#!/usr/bin/env python3
"""
针对真实数据做小范围模型诊断实验。

目标：
1. 在基线 rigid_body + friction 上增加每关节常值 bias；
2. 再增加零速保持项，观察近静止窗口误差是否明显下降；
3. 单独在近静止窗口上标定 q 零位偏置与重力向量，检查静态误差是否更像“姿态/安装”问题。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from identify_parameters import ParameterIdentifier
from load_model import URDFLoader
from run_pipeline import _split_dataframe_for_learning


@dataclass
class ExperimentRun:
    name: str
    model_options: dict[str, Any]
    identifier: ParameterIdentifier
    result: dict[str, Any]


def _load_robot_model(urdf_path: Path, gravity_vector: list[float]):
    loader = URDFLoader(str(urdf_path), gravity_vector=gravity_vector)
    return loader.build_robot_model()


def _evaluate_dataframe(df: pd.DataFrame, identifier: ParameterIdentifier, result: dict[str, Any]) -> dict[str, Any]:
    tau_meas = np.column_stack([df[f"tau_{i}"].to_numpy() for i in range(1, identifier.num_joints + 1)])
    tau_pred = identifier.predict_torques(df, result)
    err = tau_meas - tau_pred
    return {
        "rows": int(len(df)),
        "global_rmse": float(np.sqrt(np.mean(err ** 2))),
        "global_mae": float(np.mean(np.abs(err))),
        "joint_mean_err": np.round(err.mean(axis=0), 6).tolist(),
        "joint_rmse": np.round(np.sqrt(np.mean(err ** 2, axis=0)), 6).tolist(),
        "joint_mae": np.round(np.mean(np.abs(err), axis=0), 6).tolist(),
        "tau_meas_mean": np.round(tau_meas.mean(axis=0), 6).tolist(),
        "tau_pred_mean": np.round(tau_pred.mean(axis=0), 6).tolist(),
    }


def _prepare_df_for_overrides(
    df: pd.DataFrame,
    q_offset: np.ndarray | None = None,
    gravity_vector: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = df.copy()
    if q_offset is not None:
        for joint_idx in range(1, len(q_offset) + 1):
            frame[f"q_{joint_idx}"] = frame[f"q_{joint_idx}"] + float(q_offset[joint_idx - 1])
    if gravity_vector is not None:
        frame["gravity_x"] = float(gravity_vector[0])
        frame["gravity_y"] = float(gravity_vector[1])
        frame["gravity_z"] = float(gravity_vector[2])
    return frame


def _predict_with_overrides(
    df: pd.DataFrame,
    identifier: ParameterIdentifier,
    result: dict[str, Any],
    q_offset: np.ndarray,
    gravity_vector: np.ndarray,
) -> np.ndarray:
    override_df = _prepare_df_for_overrides(df, q_offset=q_offset, gravity_vector=gravity_vector)
    return identifier.predict_torques(override_df, result)


def _evaluate_with_overrides(
    df: pd.DataFrame,
    identifier: ParameterIdentifier,
    result: dict[str, Any],
    q_offset: np.ndarray,
    gravity_vector: np.ndarray,
) -> dict[str, Any]:
    tau_meas = np.column_stack([df[f"tau_{i}"].to_numpy() for i in range(1, identifier.num_joints + 1)])
    tau_pred = _predict_with_overrides(df, identifier, result, q_offset, gravity_vector)
    err = tau_meas - tau_pred
    return {
        "rows": int(len(df)),
        "global_rmse": float(np.sqrt(np.mean(err ** 2))),
        "global_mae": float(np.mean(np.abs(err))),
        "joint_mean_err": np.round(err.mean(axis=0), 6).tolist(),
        "joint_rmse": np.round(np.sqrt(np.mean(err ** 2, axis=0)), 6).tolist(),
        "joint_mae": np.round(np.mean(np.abs(err), axis=0), 6).tolist(),
        "tau_meas_mean": np.round(tau_meas.mean(axis=0), 6).tolist(),
        "tau_pred_mean": np.round(tau_pred.mean(axis=0), 6).tolist(),
    }


def _fit_q_offset_and_gravity(
    calibration_df: pd.DataFrame,
    identifier: ParameterIdentifier,
    result: dict[str, Any],
    initial_gravity: np.ndarray,
    q_offset_reg_rad: float,
    gravity_reg_scale: float,
):
    tau_meas = np.column_stack([calibration_df[f"tau_{i}"].to_numpy() for i in range(1, identifier.num_joints + 1)])
    num_joints = identifier.num_joints

    def residual_vector(params: np.ndarray) -> np.ndarray:
        q_offset = params[:num_joints]
        gravity_vector = params[num_joints:]
        tau_pred = _predict_with_overrides(
            calibration_df,
            identifier,
            result,
            q_offset=q_offset,
            gravity_vector=gravity_vector,
        )
        residual = (tau_meas - tau_pred).reshape(-1)
        regularization = np.concatenate([
            q_offset / q_offset_reg_rad,
            (gravity_vector - initial_gravity) / gravity_reg_scale,
            np.array([(np.linalg.norm(gravity_vector) - np.linalg.norm(initial_gravity)) / gravity_reg_scale]),
        ])
        return np.concatenate([residual, regularization])

    x0 = np.concatenate([np.zeros(num_joints, dtype=float), initial_gravity.astype(float)])
    solution = least_squares(residual_vector, x0=x0, method="trf")
    q_offset = solution.x[:num_joints]
    gravity_vector = solution.x[num_joints:]
    return {
        "optimizer_status": int(solution.status),
        "optimizer_message": solution.message,
        "cost": float(solution.cost),
        "success": bool(solution.success),
        "q_offset_rad": q_offset,
        "q_offset_deg": np.rad2deg(q_offset),
        "gravity_vector": gravity_vector,
    }


def _train_experiment(
    robot_model,
    train_df: pd.DataFrame,
    model_options: dict[str, Any],
    parameterization: str,
    ridge_lambda: float,
    name: str,
) -> ExperimentRun:
    identifier = ParameterIdentifier(
        robot_model,
        parameterization=parameterization,
        model_options=model_options,
    )
    result = identifier.identify_parameters(
        train_df,
        method="ridge",
        reference_parameters=None,
        ridge_lambda=ridge_lambda,
    )
    return ExperimentRun(
        name=name,
        model_options=model_options,
        identifier=identifier,
        result=result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small model-extension diagnostics on real torque data.")
    parser.add_argument("--processed-path", default=str(project_root / "datasets/real/processed/real_sensed_combined_processed.csv"))
    parser.add_argument("--focus-path", default=str(project_root / "datasets/real/normalized/vel_20_sensed_normalized.csv"))
    parser.add_argument("--focus-rows", type=int, default=50)
    parser.add_argument("--urdf-path", default=str(project_root / "models/05_urdf/urdf/05_urdf_temp.urdf"))
    parser.add_argument("--gravity", default="9.81,0,0")
    parser.add_argument("--parameterization", default="base", choices=["base", "full"])
    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--hold-eps", type=float, default=0.02, help="Zero-speed hold activation threshold in rad/s.")
    parser.add_argument("--hold-model", default="indicator", choices=["indicator", "stribeck"])
    parser.add_argument("--stribeck-scale", type=float, default=0.05, help="Velocity scale for smooth Stribeck-style hold term.")
    parser.add_argument("--q-offset-reg-deg", type=float, default=5.0)
    parser.add_argument("--gravity-reg-scale", type=float, default=1.0)
    parser.add_argument("--output-json", default=str(project_root / "output/diagnostics/model_extension_experiments.json"))
    args = parser.parse_args()

    processed_df = pd.read_csv(args.processed_path)
    focus_df = pd.read_csv(args.focus_path).head(args.focus_rows).reset_index(drop=True)

    gravity_vector = np.array([float(part.strip()) for part in args.gravity.split(",")], dtype=float)
    robot_model = _load_robot_model(Path(args.urdf_path), gravity_vector.tolist())
    splits = _split_dataframe_for_learning(processed_df, seed=123)

    experiments = [
        ("baseline", {}),
        ("bias", {"include_bias": True}),
        (
            "bias_hold",
            {
                "include_bias": True,
                "include_hold": True,
                "hold_model": args.hold_model,
                "hold_velocity_epsilon": args.hold_eps,
                "stribeck_velocity_scale": args.stribeck_scale,
            },
        ),
    ]

    experiment_runs: list[ExperimentRun] = []
    for name, model_options in experiments:
        print(f"\n=== Training experiment: {name} ===")
        experiment_runs.append(
            _train_experiment(
                robot_model=robot_model,
                train_df=splits["train"],
                model_options=model_options,
                parameterization=args.parameterization,
                ridge_lambda=args.ridge_lambda,
                name=name,
            )
        )

    report: dict[str, Any] = {
        "focus_path": str(args.focus_path),
        "focus_rows": int(args.focus_rows),
        "processed_path": str(args.processed_path),
        "gravity_vector_initial": gravity_vector.tolist(),
        "experiments": {},
    }

    for run in experiment_runs:
        print(f"\n=== Evaluating experiment: {run.name} ===")
        report["experiments"][run.name] = {
            "model_options": run.model_options,
            "train_fit": {
                "rank": int(run.result["rank"]),
                "full_regressor_rank": int(run.result["full_regressor_rank"]),
                "condition_number": float(run.result["condition_number"]),
                "active_condition_number": float(run.result["active_condition_number"]),
                "num_parameters_full": int(run.result["num_parameters_full"]),
                "num_parameters_active": int(run.result["num_parameters_active"]),
            },
            "metrics": {
                "focus_first_rows": _evaluate_dataframe(focus_df, run.identifier, run.result),
                "train": _evaluate_dataframe(splits["train"], run.identifier, run.result),
                "val": _evaluate_dataframe(splits["val"], run.identifier, run.result),
                "test": _evaluate_dataframe(splits["test"], run.identifier, run.result),
                "full_processed": _evaluate_dataframe(processed_df, run.identifier, run.result),
            },
        }

    print("\n=== Calibrating q-offset + gravity on focus window (baseline model fixed) ===")
    baseline_run = experiment_runs[0]
    calibration = _fit_q_offset_and_gravity(
        calibration_df=focus_df,
        identifier=baseline_run.identifier,
        result=baseline_run.result,
        initial_gravity=gravity_vector,
        q_offset_reg_rad=np.deg2rad(args.q_offset_reg_deg),
        gravity_reg_scale=args.gravity_reg_scale,
    )
    report["q_offset_gravity_calibration"] = {
        "success": calibration["success"],
        "optimizer_status": calibration["optimizer_status"],
        "optimizer_message": calibration["optimizer_message"],
        "cost": calibration["cost"],
        "q_offset_rad": np.round(calibration["q_offset_rad"], 8).tolist(),
        "q_offset_deg": np.round(calibration["q_offset_deg"], 6).tolist(),
        "gravity_vector": np.round(calibration["gravity_vector"], 6).tolist(),
        "metrics": {
            "focus_first_rows_before": _evaluate_dataframe(focus_df, baseline_run.identifier, baseline_run.result),
            "focus_first_rows_after": _evaluate_with_overrides(
                focus_df,
                baseline_run.identifier,
                baseline_run.result,
                q_offset=calibration["q_offset_rad"],
                gravity_vector=calibration["gravity_vector"],
            ),
            "full_processed_after": _evaluate_with_overrides(
                processed_df,
                baseline_run.identifier,
                baseline_run.result,
                q_offset=calibration["q_offset_rad"],
                gravity_vector=calibration["gravity_vector"],
            ),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved experiment report to: {output_path}")
    print("\nFocus-window summary:")
    for name in ("baseline", "bias", "bias_hold"):
        metrics = report["experiments"][name]["metrics"]["focus_first_rows"]
        print(
            f"  {name:<10} RMSE={metrics['global_rmse']:.6f}  "
            f"MAE={metrics['global_mae']:.6f}"
        )
    calibrated = report["q_offset_gravity_calibration"]["metrics"]["focus_first_rows_after"]
    print(
        f"  q+g calib  RMSE={calibrated['global_rmse']:.6f}  "
        f"MAE={calibrated['global_mae']:.6f}"
    )


if __name__ == "__main__":
    main()
