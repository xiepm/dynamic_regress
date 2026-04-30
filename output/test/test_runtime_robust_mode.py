#!/usr/bin/env python3
"""
Compile and exercise the generated robust runtime mode in C++.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED_CPP = PROJECT_ROOT / "output" / "sevendofDynamics.cpp"
GENERATED_HEADER = GENERATED_CPP.with_suffix(".h")


HARNESS_SOURCE = r"""
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "sevendofDynamics.h"

namespace
{
constexpr EcSizeT kNumJoints = 7;

bool allClose(const EcRealVector& lhs, const EcRealVector& rhs, const double tol)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (EcSizeT idx = 0; idx < lhs.size(); ++idx)
    {
        if (std::fabs(lhs[idx] - rhs[idx]) > tol)
        {
            return false;
        }
    }
    return true;
}

void expect(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}
}  // namespace

int main()
{
    sevendofDynamics dynamics;
    EcRealVector parms;
    EcRealVector q(kNumJoints, 0.0);
    EcRealVector dq(kNumJoints, 0.0);
    EcRealVector ddq(kNumJoints, 0.0);
    EcRealVector tauLegacyA;
    EcRealVector tauLegacyB;

    expect(dynamics.calculateEstimateJointToqrues(q, dq, ddq, parms, tauLegacyA), "legacy call should succeed");
    dynamics.setRuntimeTorqueMode(LEGACY_RAW);
    expect(dynamics.calculateEstimateJointToqrues(q, dq, ddq, parms, tauLegacyB), "explicit LEGACY_RAW call should succeed");
    expect(allClose(tauLegacyA, tauLegacyB, 1e-12), "legacy compatibility mismatch");

    RuntimeNoTimeConfig runtimeCfg;
    runtimeCfg.enableInputValidation = true;
    runtimeCfg.enableSampleSmoothing = false;
    runtimeCfg.enableDdqClamp = true;
    runtimeCfg.enableFrictionStateMachine = true;
    runtimeCfg.enableHoldMode = true;
    runtimeCfg.enableMCGate = true;
    dynamics.setRuntimeNoTimeConfig(runtimeCfg);

    DdqClampConfig clampCfg;
    clampCfg.enabled = true;
    clampCfg.ddqLimit.assign(kNumJoints, 1.0);
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

    dynamics.setRuntimeTorqueMode(ROBUST_NO_TIME);
    dynamics.resetRuntimeRobustState();

    RuntimeDiagnostics diagnostics;
    EcRealVector tauRobust;

    dq.assign(kNumJoints, 0.0);
    ddq.assign(kNumJoints, 0.0);
    dq[0] = 0.05;
    expect(dynamics.calculateEstimateJointToqrues(q, dq, ddq, parms, tauRobust), "positive-motion robust call should succeed");
    dynamics.getLastRuntimeDiagnostics(diagnostics);
    expect(static_cast<int>(diagnostics.motion_state[0]) == MOVING_POSITIVE, "expected MOVING_POSITIVE");
    expect(std::fabs(diagnostics.friction_sign[0] - 1.0) < 1e-12, "expected positive friction sign");

    dq[0] = -0.05;
    expect(dynamics.calculateEstimateJointToqrues(q, dq, ddq, parms, tauRobust), "negative-motion robust call should succeed");
    dynamics.getLastRuntimeDiagnostics(diagnostics);
    expect(static_cast<int>(diagnostics.motion_state[0]) == MOVING_NEGATIVE, "expected MOVING_NEGATIVE");
    expect(std::fabs(diagnostics.friction_sign[0] + 1.0) < 1e-12, "expected negative friction sign");

    dynamics.resetRuntimeRobustState();
    int signChanges = 0;
    double lastSign = 0.0;
    for (int sampleIdx = 0; sampleIdx < 12; ++sampleIdx)
    {
        dq.assign(kNumJoints, 0.0);
        ddq.assign(kNumJoints, 0.01);
        dq[0] = (sampleIdx % 2 == 0) ? 0.005 : -0.005;
        expect(dynamics.calculateEstimateJointToqrues(q, dq, ddq, parms, tauRobust), "static-jitter robust call should succeed");
        dynamics.getLastRuntimeDiagnostics(diagnostics);
        if (sampleIdx > 0 && std::fabs(diagnostics.friction_sign[0] - lastSign) > 1e-12)
        {
            ++signChanges;
        }
        lastSign = diagnostics.friction_sign[0];
        expect(std::fabs(diagnostics.wM[0]) < 1e-12, "static jitter should gate tau_M");
        expect(std::fabs(diagnostics.wC[0]) < 1e-12, "static jitter should gate tau_C");
    }
    expect(signChanges == 0, "friction sign should not chatter in static jitter test");

    dynamics.resetRuntimeRobustState();
    dq.assign(kNumJoints, 0.0);
    ddq.assign(kNumJoints, 0.0);
    ddq[0] = 100.0;
    expect(dynamics.calculateEstimateJointToqrues(q, dq, ddq, parms, tauRobust), "ddq spike robust call should succeed");
    dynamics.getLastRuntimeDiagnostics(diagnostics);
    expect(std::fabs(diagnostics.ddq_used[0] - 1.0) < 1e-12, "ddq clamp did not trigger");

    q.assign(kNumJoints, 180.0);
    dq.assign(kNumJoints, 0.0);
    ddq.assign(kNumJoints, 0.0);
    expect(!dynamics.calculateEstimateJointToqrues(q, dq, ddq, parms, tauRobust), "degree input should fail validation");
    expect(allClose(tauRobust, EcRealVector(kNumJoints, 0.0), 1e-12), "failed validation must zero tau");

    std::cout << "PASSED: runtime robust mode checks\n";
    return 0;
}
"""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="runtime_robust_test_", dir=str(PROJECT_ROOT / "output" / "test")) as tmpdir:
        tmpdir_path = Path(tmpdir)
        harness_cpp = tmpdir_path / "runtime_robust_harness.cpp"
        binary = tmpdir_path / "runtime_robust_harness"
        harness_cpp.write_text(HARNESS_SOURCE, encoding="utf-8")

        command = [
            "g++",
            "-std=c++17",
            "-O2",
            "-I",
            str(PROJECT_ROOT / "output" / "test" / "rtos_stub"),
            "-I",
            str(GENERATED_CPP.parent),
            str(GENERATED_CPP),
            str(harness_cpp),
            "-o",
            str(binary),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        result = subprocess.run([str(binary)], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
        print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
