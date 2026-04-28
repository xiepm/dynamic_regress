#pragma once

#include "hansTypes.h"

class dynamicsBase
{
public:
    virtual ~dynamicsBase() = default;

    virtual void setRobotDHParameters(const EcRealVector& kinematcisParam) = 0;
    virtual void setGravityVector(const EcReal gx, const EcReal gy, const EcReal gz) = 0;
    virtual void calculateGravityJointTorques(
        const EcRealVector& q,
        const EcRealVector& parms,
        EcRealVector& tau
    ) = 0;
    virtual EcBoolean calculateEstimateJointToqrues(
        const EcRealVector& q,
        const EcRealVector& dq,
        const EcRealVector& ddq,
        const EcRealVector& parms,
        EcRealVector& tau
    ) = 0;
    virtual void calculateMomentumEstimatedJointTorques(
        const EcRealVector& q,
        const EcRealVector& dq,
        const EcRealVector& ddq,
        const EcRealVector& parms,
        EcRealVector& tau
    ) = 0;
};
