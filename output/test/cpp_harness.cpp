#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef GENERATED_DYNAMICS_HEADER
#define GENERATED_DYNAMICS_HEADER "identifiedDynamics.h"
#endif

#ifndef GENERATED_DYNAMICS_CLASS
#define GENERATED_DYNAMICS_CLASS identifiedDynamics
#endif

#include GENERATED_DYNAMICS_HEADER

namespace
{
constexpr EcSizeT kNumJoints = 7;

void readJointVector(std::istream& input, EcRealVector& values)
{
    values.assign(kNumJoints, 0.0);
    for (EcSizeT idx = 0; idx < kNumJoints; ++idx)
    {
        if (!(input >> values[idx]))
        {
            throw std::runtime_error("Failed to read expected joint value from input sample file.");
        }
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <gx> <gy> <gz>\n";
        return 2;
    }

    const std::string input_path = argv[1];
    const EcReal gx = std::stod(argv[2]);
    const EcReal gy = std::stod(argv[3]);
    const EcReal gz = std::stod(argv[4]);

    std::ifstream input(input_path);
    if (!input.is_open())
    {
        std::cerr << "Failed to open input file: " << input_path << "\n";
        return 3;
    }

    GENERATED_DYNAMICS_CLASS dynamics;
    dynamics.setGravityVector(gx, gy, gz);

    EcRealVector q;
    EcRealVector dq;
    EcRealVector ddq;
    EcRealVector tau;
    EcRealVector tauM;
    EcRealVector tauC;
    EcRealVector tauGxUnit;
    EcRealVector tauGyUnit;
    EcRealVector tauGzUnit;
    EcRealVector tauGravity;
    EcRealVector tauFriction;
    EcRealVector parms;

    std::cout << std::setprecision(17);
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::istringstream line_stream(line);
        readJointVector(line_stream, q);
        readJointVector(line_stream, dq);
        readJointVector(line_stream, ddq);

        if (!dynamics.calculateEstimateJointToqrues(q, dq, ddq, parms, tau))
        {
            std::cerr << "calculateEstimateJointToqrues returned false.\n";
            return 4;
        }
        dynamics.computeMassTorque(q, ddq, tauM);
        dynamics.computeCoriolisTorque(q, dq, tauC);
        dynamics.computeGravityAxisTorques(q, tauGxUnit, tauGyUnit, tauGzUnit);
        dynamics.calculateGravityJointTorques(q, parms, tauGravity);
        dynamics.computeFrictionTorque(q, dq, ddq, tauFriction);

        auto printNamedVector = [](const char* name, const EcRealVector& values)
        {
            std::cout << name;
            for (EcSizeT idx = 0; idx < values.size(); ++idx)
            {
                std::cout << (idx == 0 ? ' ' : ' ') << values[idx];
            }
            std::cout << '\n';
        };

        printNamedVector("tau_M", tauM);
        printNamedVector("tau_C", tauC);
        printNamedVector("tau_Gx_unit", tauGxUnit);
        printNamedVector("tau_Gy_unit", tauGyUnit);
        printNamedVector("tau_Gz_unit", tauGzUnit);
        printNamedVector("tau_gravity", tauGravity);
        printNamedVector("tau_friction", tauFriction);
        printNamedVector("tau_pred", tau);
    }

    return 0;
}
